"""Pull seasons of MLB data into a local SQLite database.

Two tables:
  games          — one row per completed regular-season game
  pitcher_starts — one row per start, per pitcher (earned runs + outs)

Design rule: this module only RECORDS what happened and when. All
point-in-time logic lives in features.py, so leakage bugs have exactly
one place to hide.
"""
from __future__ import annotations

import json
import sqlite3
import time
import urllib.request
from pathlib import Path

API = "https://statsapi.mlb.com/api/v1"

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
  game_pk   INTEGER PRIMARY KEY,
  date      TEXT NOT NULL,
  season    INTEGER NOT NULL,
  away_id   INTEGER NOT NULL,
  home_id   INTEGER NOT NULL,
  away_score INTEGER NOT NULL,
  home_score INTEGER NOT NULL,
  away_sp   INTEGER,
  home_sp   INTEGER
);
CREATE INDEX IF NOT EXISTS idx_games_date ON games(date);

CREATE TABLE IF NOT EXISTS pitcher_starts (
  pitcher_id INTEGER NOT NULL,
  date       TEXT NOT NULL,
  season     INTEGER NOT NULL,
  er         INTEGER NOT NULL,
  outs       INTEGER NOT NULL,
  so         INTEGER,
  bb         INTEGER,
  bf         INTEGER,
  PRIMARY KEY (pitcher_id, date)
);
CREATE INDEX IF NOT EXISTS idx_ps_pid ON pitcher_starts(pitcher_id, date);

CREATE TABLE IF NOT EXISTS batter_games (
  batter_id INTEGER NOT NULL,
  date      TEXT NOT NULL,
  season    INTEGER NOT NULL,
  team_id   INTEGER,
  name      TEXT,
  pa        INTEGER,
  ab        INTEGER,
  h         INTEGER,
  hr        INTEGER,
  PRIMARY KEY (batter_id, date)
);
CREATE INDEX IF NOT EXISTS idx_bg_bid ON batter_games(batter_id, date);
CREATE INDEX IF NOT EXISTS idx_bg_team ON batter_games(team_id, date);
"""


def _get(url: str, retries: int = 3, pause: float = 1.0) -> dict:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.loads(r.read().decode())
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(pause * (attempt + 1))
    raise RuntimeError("unreachable")


def _ip_to_outs(ip) -> int:
    whole, _, frac = str(ip or "0").partition(".")
    return int(whole or 0) * 3 + int(frac or 0)


def connect(db_path: str | Path) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    cols = {r[1] for r in con.execute("PRAGMA table_info(pitcher_starts)")}
    for c in ("so", "bb", "bf"):
        if c not in cols:
            con.execute(f"ALTER TABLE pitcher_starts ADD COLUMN {c} INTEGER")
    con.commit()
    return con


def ingest_batting(con: sqlite3.Connection, season: int,
                   verbose: bool = True) -> None:
    """Hitting game logs for every position player on a season roster.

    Same per-player gameLog pattern as the pitcher ingest — one polite
    call per batter. ~1,000 players per season, so this is the slow
    half of a bootstrap; nightly incremental runs only touch the
    current season.
    """
    teams = _get(f"{API}/teams?sportId=1&season={season}")
    bat_ids: dict[int, int] = {}
    for t in teams.get("teams", []):
        roster = _get(f"{API}/teams/{t['id']}/roster"
                      f"?rosterType=fullSeason&season={season}")
        for r in roster.get("roster", []):
            if (r.get("position") or {}).get("type") != "Pitcher":
                pid = (r.get("person") or {}).get("id")
                if pid:
                    bat_ids[pid] = t["id"]
        time.sleep(0.1)
    if verbose:
        print(f"  {season}: {len(bat_ids)} position players on rosters")
    for n, (pid, team_id) in enumerate(sorted(bat_ids.items()), 1):
        j = _get(f"{API}/people/{pid}"
                 f"?hydrate=stats(group=[hitting],type=[gameLog],season={season})")
        person = (j.get("people") or [{}])[0]
        name = person.get("fullName")
        blocks = person.get("stats") or []
        log = next((b for b in blocks
                    if b.get("type", {}).get("displayName") == "gameLog"), None)
        rows = []
        for sp in (log or {}).get("splits", []):
            st = sp.get("stat", {})
            pa = int(st.get("plateAppearances") or 0)
            if pa == 0:
                continue
            rows.append((pid, sp["date"], season,
                         (sp.get("team") or {}).get("id") or team_id, name,
                         pa, int(st.get("atBats") or 0),
                         int(st.get("hits") or 0),
                         int(st.get("homeRuns") or 0)))
        con.executemany(
            "INSERT OR REPLACE INTO batter_games VALUES (?,?,?,?,?,?,?,?,?)",
            rows)
        if verbose and n % 100 == 0:
            print(f"  ...batter logs {n}/{len(bat_ids)}")
        con.commit()
        time.sleep(0.12)


def ingest_season(con: sqlite3.Connection, season: int, verbose: bool = True) -> None:
    """Load one season's completed games and the game logs of every starter."""
    sched = _get(
        f"{API}/schedule?sportId=1&startDate={season}-01-01&endDate={season}-12-31"
        f"&gameType=R&hydrate=probablePitcher"
    )
    rows, sp_ids = [], set()
    for d in sched.get("dates", []):
        for g in d.get("games", []):
            if g.get("status", {}).get("abstractGameState") != "Final":
                continue
            a, h = g["teams"]["away"], g["teams"]["home"]
            if a.get("score") is None or h.get("score") is None:
                continue
            if a["score"] == h["score"]:
                continue
            asp = (a.get("probablePitcher") or {}).get("id")
            hsp = (h.get("probablePitcher") or {}).get("id")
            rows.append((g["gamePk"], d["date"], season, a["team"]["id"],
                         h["team"]["id"], a["score"], h["score"], asp, hsp))
            sp_ids.update(i for i in (asp, hsp) if i)

    con.executemany(
        "INSERT OR REPLACE INTO games VALUES (?,?,?,?,?,?,?,?,?)", rows
    )
    con.commit()
    if verbose:
        print(f"  {season}: {len(rows)} games, {len(sp_ids)} starters")

    for n, pid in enumerate(sorted(sp_ids), 1):
        j = _get(
            f"{API}/people/{pid}"
            f"?hydrate=stats(group=[pitching],type=[gameLog],season={season})"
        )
        blocks = (j.get("people") or [{}])[0].get("stats") or []
        log = next((b for b in blocks
                    if b.get("type", {}).get("displayName") == "gameLog"), None)
        starts = []
        for s in (log or {}).get("splits", []):
            st = s.get("stat", {})
            if int(st.get("gamesStarted") or 0) != 1:
                continue
            starts.append((pid, s["date"], season,
                           int(st.get("earnedRuns") or 0),
                           _ip_to_outs(st.get("inningsPitched")),
                           int(st.get("strikeOuts") or 0),
                           int(st.get("baseOnBalls") or 0),
                           int(st.get("battersFaced") or 0)))
        con.executemany(
            "INSERT OR REPLACE INTO pitcher_starts VALUES (?,?,?,?,?,?,?,?)",
            starts
        )
        if verbose and n % 50 == 0:
            print(f"  ...pitcher logs {n}/{len(sp_ids)}")
        con.commit()
        time.sleep(0.15)  # be polite to a free public API

    ingest_batting(con, season, verbose)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Ingest MLB seasons into SQLite")
    p.add_argument("--db", default="slate.db")
    p.add_argument("--seasons", type=int, nargs="+", required=True)
    args = p.parse_args()
    con = connect(args.db)
    for season in args.seasons:
        print(f"Ingesting {season}…")
        ingest_season(con, season)
    n = con.execute("SELECT COUNT(*) FROM games").fetchone()[0]
    print(f"Done. {n} games in {args.db}")


if __name__ == "__main__":
    main()
