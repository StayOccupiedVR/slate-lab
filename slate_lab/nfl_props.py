"""NFL player props v1 — yardage distributions. Validated 2026-08-12.

SHIPS   Receiving yards (WR/TE/RB) and rushing yards (RB, 5+ carry
        games): prior-blended empirical distributions. Player pool =
        last YD_WIN games; position baseline mixed in at a weight
        worth YARD_PRIOR_GAMES games. Quantiles + P(over line) come
        straight off the pool.

        Walk-forward quantile coverage (targets 25/50/75%):
            rec_yards  2023 28/54/77   2024 26/51/75   2025 28/53/78
            rush_yards 2023 29/54/78   2024 28/49/72   2025 26/52/78
        Within ~3pts everywhere; q25 runs ~2-3 high (left tail —
        injury/blowout near-zero games — slightly heavier in reality
        than the pool). Documented, acceptable for research display.

BENCHED Receptions. Three formulations, none shippable:
        - Binomial mixture on career targets: log 2.00-2.07 vs naive
          Poisson 1.92-1.97 (WORSE, all seasons).
        - Same with 16-game recency: 2.06-2.13 (worse still).
        - Poisson on recent mean (the naive itself): best log score
          (1.91-1.96) but 5-8pts HOT above 45% (50->41, 70->62,
          88->79 across seasons) — receptions are overdispersed vs
          Poisson because target volume itself swings.
        v2 path: negative binomial / target-count mixture with a
        dispersion parameter. Until that passes, no receptions market.

LESSON  The inverse of the MLB workload finding, worth remembering:
        pitcher workloads are stable (full career won); NFL roles are
        coaching decisions that change monthly (recency won for
        yardage, and role volatility is exactly what sank receptions).

DATA    nflverse weekly player stats parquet, 2020+. Asset name moved
        mid-2025: try player_stats/player_stats_{s}.parquet then
        stats_player/stats_player_week_{s}.parquet.
"""
from __future__ import annotations

import argparse
import io
import sqlite3
import urllib.request
from collections import defaultdict

import numpy as np
import pandas as pd

MIN_GAMES = 8
YD_WIN = 24              # player pool: last N games (recency ablation win)
YARD_PRIOR_GAMES = 12    # position baseline weight, in game-equivalents
QS = (0.10, 0.25, 0.50, 0.75, 0.90)

SCHEMA = """
CREATE TABLE IF NOT EXISTS nfl_player_weeks (
  player_id TEXT NOT NULL,
  season    INTEGER NOT NULL,
  week      INTEGER NOT NULL,
  name      TEXT,
  position  TEXT,
  team      TEXT,
  targets   REAL,
  receptions REAL,
  rec_yards REAL,
  carries   REAL,
  rush_yards REAL,
  PRIMARY KEY (player_id, season, week)
);
CREATE INDEX IF NOT EXISTS idx_npw ON nfl_player_weeks(player_id, season, week);
"""

_URLS = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "player_stats/player_stats_{s}.parquet",
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "stats_player/stats_player_week_{s}.parquet",
)


def ingest_player_weeks(con, season: int, verbose: bool = True) -> None:
    con.executescript(SCHEMA)
    df = None
    for url in _URLS:
        try:
            raw = urllib.request.urlopen(url.format(s=season), timeout=120).read()
            df = pd.read_parquet(io.BytesIO(raw))
            break
        except Exception:
            continue
    if df is None:
        print(f"  {season}: no player stats asset found")
        return
    df = df[df.season_type == "REG"]
    df = df[df.player_id.notna()]      # 2025 file carries a few empty rows
    team_col = "recent_team" if "recent_team" in df.columns else "team"
    rows = [(r.player_id, int(r.season), int(r.week),
             getattr(r, "player_display_name", None), r.position,
             getattr(r, team_col, None),
             r.targets, r.receptions, r.receiving_yards,
             r.carries, r.rushing_yards)
            for r in df.itertuples()]
    con.executemany(
        "INSERT OR REPLACE INTO nfl_player_weeks VALUES "
        "(?,?,?,?,?,?,?,?,?,?,?)", rows)
    con.commit()
    if verbose:
        print(f"  {season}: {len(rows)} player-weeks stored")


def yards_distribution(vals: list[float], pos_vals: np.ndarray,
                       rng=None) -> dict | None:
    """Prior-blended empirical yardage distribution.

    vals = the player's per-game yards (chronological); pos_vals = the
    position baseline pool. Returns quantiles and a callable-free pool
    summary the feed can serialize.
    """
    vals = [v for v in vals if v is not None][-YD_WIN:]
    if len(vals) < MIN_GAMES or pos_vals is None or not len(pos_vals):
        return None
    rng = rng or np.random.default_rng(0)
    n = len(vals)
    w = YARD_PRIOR_GAMES / (n + YARD_PRIOR_GAMES)
    k = max(1, int(n * w / (1 - w)))
    pool = np.concatenate([np.asarray(vals, dtype=float),
                           rng.choice(pos_vals, size=k, replace=True)])
    q = np.quantile(pool, QS)
    return {
        "q": {str(int(p * 100)): round(float(v), 1) for p, v in zip(QS, q)},
        "mean": round(float(pool.mean()), 1),
        "n_prior": n,
        "over": {str(line): round(float((pool > line).mean()), 3)
                 for line in (25, 40, 50, 60, 75, 100)},
    }


def _pools(con, before_season: int, before_week: int | None = None):
    q = ("SELECT * FROM nfl_player_weeks WHERE season < ? "
         "ORDER BY season, week")
    df = pd.read_sql(q, con, params=(before_season,))
    if before_week is not None:
        cur = pd.read_sql(
            "SELECT * FROM nfl_player_weeks WHERE season = ? AND week < ? "
            "ORDER BY week", con, params=(before_season, before_week))
        df = pd.concat([df, cur])
    hist = defaultdict(list)
    pos_ry = defaultdict(list)
    pos_rush = []
    names = {}
    for r in df.itertuples():
        hist[r.player_id].append(r)
        names[r.player_id] = r.name
        if r.position in ("WR", "TE", "RB") and r.targets and r.targets > 0:
            pos_ry[r.position].append(r.rec_yards)
        if r.position == "RB" and r.carries and r.carries >= 5:
            pos_rush.append(r.rush_yards)
    return hist, pos_ry, pos_rush, names


def project_week(con, season: int, week: int) -> dict:
    """Yardage distributions for the players likely to feature in each
    game of a week. 'Likely' = usage (targets+carries) in the last 5
    played weeks; top 8 per team. Pure function of the two tables."""
    sched = pd.read_sql(
        "SELECT game_id, away_team, home_team, gameday FROM nfl_games "
        "WHERE season=? AND week=? AND game_type='REG' ORDER BY gameday",
        con, params=(season, week))
    hist, pos_ry, pos_rush, names = _pools(con, season, week)
    # usage in this season's recent weeks
    recent = pd.read_sql(
        "SELECT player_id, team, position, "
        "SUM(COALESCE(targets,0)+COALESCE(carries,0)) AS usage_ct, "
        "COUNT(*) AS g FROM nfl_player_weeks "
        "WHERE season=? AND week>=? AND week<? GROUP BY player_id",
        con, params=(season, max(1, week - 5), week))
    per_team = defaultdict(list)
    for r in recent.itertuples():
        if r.team and r.usage_ct and r.usage_ct > 0:
            per_team[r.team].append((r.player_id, r.position, r.usage_ct))
    games = []
    for g in sched.itertuples():
        players = []
        for team, opp in ((g.away_team, g.home_team),
                          (g.home_team, g.away_team)):
            cand = sorted(per_team.get(team, []), key=lambda x: -x[2])[:8]
            for pid, pos, _u in cand:
                pr = hist.get(pid, [])
                entry = {"player_id": pid, "name": names.get(pid) or pid,
                         "pos": pos, "team": team, "opp": opp}
                if pos in ("WR", "TE", "RB"):
                    d = yards_distribution(
                        [x.rec_yards for x in pr
                         if x.targets and x.targets > 0],
                        np.array(pos_ry.get(pos, [])))
                    if d:
                        entry["rec_yards"] = d
                if pos == "RB":
                    d = yards_distribution(
                        [x.rush_yards for x in pr
                         if x.carries and x.carries >= 5],
                        np.array(pos_rush))
                    if d:
                        entry["rush_yards"] = d
                if "rec_yards" in entry or "rush_yards" in entry:
                    players.append(entry)
        games.append({"game_id": g.game_id, "away": g.away_team,
                      "home": g.home_team, "gameday": g.gameday,
                      "players": players})
    return {"season": season, "week": week, "market": "yardage",
            "model": "nfl-props-v1", "games": games}


def backtest(con, season: int) -> None:
    hist, pos_ry, pos_rush, _ = _pools(con, season)
    te = pd.read_sql(
        "SELECT * FROM nfl_player_weeks WHERE season = ? ORDER BY week",
        con, params=(season,))
    for market in ("rec_yards", "rush_yards"):
        cov = {q: 0 for q in QS}
        n = 0
        h2 = {k: list(v) for k, v in hist.items()}
        pry = {k: list(v) for k, v in pos_ry.items()}
        prush = list(pos_rush)
        for r in te.itertuples():
            pr = h2.get(r.player_id, [])
            d = None
            actual = None
            if market == "rec_yards" and r.position in ("WR", "TE", "RB") \
                    and r.targets and r.targets > 0:
                d = yards_distribution(
                    [g.rec_yards for g in pr if g.targets and g.targets > 0],
                    np.array(pry.get(r.position, [])))
                actual = r.rec_yards
            elif market == "rush_yards" and r.position == "RB" \
                    and r.carries and r.carries >= 5:
                d = yards_distribution(
                    [g.rush_yards for g in pr if g.carries and g.carries >= 5],
                    np.array(prush))
                actual = r.rush_yards
            if d is not None:
                n += 1
                for p in QS:
                    if actual <= d["q"][str(int(p * 100))]:
                        cov[p] += 1
            h2.setdefault(r.player_id, []).append(r)
            if r.position in ("WR", "TE", "RB") and r.targets and r.targets > 0:
                pry.setdefault(r.position, []).append(r.rec_yards)
            if r.position == "RB" and r.carries and r.carries >= 5:
                prush.append(r.rush_yards)
        if n == 0:
            print(f"{market} {season}: no scoreable rows")
            continue
        line = " ".join(f"q{int(p*100)}={cov[p]/n:.1%}" for p in QS)
        print(f"{market} {season}: n={n} coverage {line}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="nfl.db")
    p.add_argument("--ingest", nargs="+", type=int, default=None)
    p.add_argument("--backtest", type=int, default=None)
    p.add_argument("--project", action="store_true",
                   help="project the upcoming week's players")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    con = sqlite3.connect(args.db)
    if args.ingest:
        for s in args.ingest:
            ingest_player_weeks(con, s)
    if args.backtest:
        backtest(con, args.backtest)
    if args.project:
        import json
        from .nfl_ledger import upcoming_week
        sw = upcoming_week(con)
        if sw is None:
            print("No regular-season NFL week in the window — sleeping.")
            return
        doc = project_week(con, *sw)
        out = args.out or "nfl-props.json"
        with open(out, "w") as f:
            json.dump(doc, f, indent=1)
        n = sum(len(g["players"]) for g in doc["games"])
        print(f"week {sw[1]}: {n} players across {len(doc['games'])} games -> {out}")


if __name__ == "__main__":
    main()
