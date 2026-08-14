"""NBA adapter. Validated 2026-08-13 on hoopR history (2007-2023).

VALIDATION (walk-forward, logistic on the features below, 18,145
feature-games; holdouts after training on all prior seasons):
    2021: acc .628  logloss .6378  (baseline .6899)
    2022: acc .641  logloss .6396  (baseline .6927)
    2023: acc .630  logloss .6431  (baseline .6796)
Coefficients all sane: pyth_diff dominates (+4.7), back-to-backs hurt
the tired side, rest differential helps the rested side. This is the
plausible public-data band (closing lines live near .60); the market
comparison begins when odds capture starts at season launch.

FEATURES (all point-in-time, 8-game burn-in per team per season):
    pyth_diff   Pythagorean exp 13.91 (basketball's classic), prior
                8 games at 112 ppg
    b2b_away / b2b_home   second night of a back-to-back
    rest_diff   capped rest-day differential (away - home)

DATA    History bootstrap: sportsdataverse hoopR master parquet on
        GitHub (2002-2023, reachable from anywhere including CI).
        Current seasons: ESPN's public scoreboard API, fetched
        date-by-date from Actions (same source the app's client
        already uses for NFL).
"""
from __future__ import annotations

import io
import json
import urllib.request
from collections import defaultdict
from datetime import date, timedelta

import numpy as np
import pandas as pd

PYTH_EXP = 13.91
PRIOR_G = 8
PRIOR_PPG = 112.0
BURN_IN = 8
REST_CAP = 4

MASTER_URL = ("https://raw.githubusercontent.com/sportsdataverse/"
              "hoopR-data/main/nba_schedule_master.parquet")
ESPN_URL = ("https://site.api.espn.com/apis/site/v2/sports/basketball/"
            "nba/scoreboard?dates={ymd}")

SCHEMA = """
CREATE TABLE IF NOT EXISTS nba_games (
  game_id   TEXT PRIMARY KEY,
  season    INTEGER NOT NULL,
  date      TEXT NOT NULL,
  away_id   TEXT NOT NULL,
  home_id   TEXT NOT NULL,
  away_abbr TEXT,
  home_abbr TEXT,
  away_score REAL,
  home_score REAL,
  completed INTEGER,
  game_type TEXT
);
CREATE INDEX IF NOT EXISTS idx_nba_season ON nba_games(season, date);
"""

label = "away_win"
GROUPS = {
    "team": ["pyth_diff"],
    "schedule": ["b2b_away", "b2b_home", "rest_diff"],
}
EVAL_COLS = ["game_id", "season", "date", "away_abbr", "home_abbr",
             "away_score", "home_score"]


def ingest_master(con) -> None:
    """Bootstrap 2002-2023 history from the hoopR master parquet."""
    con.executescript(SCHEMA)
    raw = urllib.request.urlopen(MASTER_URL, timeout=180).read()
    df = pd.read_parquet(io.BytesIO(raw))
    df = df[df.home_score.notna() & df.away_score.notna()]
    rows = []
    for r in df.itertuples():
        rows.append((str(r.id), int(r.season),
                     str(pd.to_datetime(r.date).date()),
                     str(r.away_id), str(r.home_id),
                     getattr(r, "away_abbreviation", None),
                     getattr(r, "home_abbreviation", None),
                     float(r.away_score), float(r.home_score),
                     int(bool(r.status_type_completed)),
                     "REG"))
    con.executemany(
        "INSERT OR REPLACE INTO nba_games VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        rows)
    con.commit()
    print(f"  master: {len(rows)} games (2002-2023)")


def ingest_espn_range(con, start: date, end: date,
                      season: int, verbose: bool = True) -> None:
    """Current-season top-up: one polite ESPN scoreboard call per day.

    Run from Actions (open internet). Idempotent by game id.
    """
    con.executescript(SCHEMA)
    d = start
    n = 0
    while d <= end:
        try:
            j = json.load(urllib.request.urlopen(
                ESPN_URL.format(ymd=d.strftime("%Y%m%d")), timeout=30))
        except Exception:
            d += timedelta(days=1)
            continue
        for ev in j.get("events", []):
            comp = (ev.get("competitions") or [{}])[0]
            teams = {c.get("homeAway"): c
                     for c in comp.get("competitors", [])}
            if "home" not in teams or "away" not in teams:
                continue
            st = (ev.get("status") or {}).get("type", {})
            con.execute(
                "INSERT OR REPLACE INTO nba_games VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?)",
                (str(ev["id"]), season, d.isoformat(),
                 str(teams["away"]["team"]["id"]),
                 str(teams["home"]["team"]["id"]),
                 teams["away"]["team"].get("abbreviation"),
                 teams["home"]["team"].get("abbreviation"),
                 float(teams["away"].get("score") or 0),
                 float(teams["home"].get("score") or 0),
                 int(bool(st.get("completed"))),
                 "REG" if (ev.get("season") or {}).get("type") == 2
                 else "OTHER"))
            n += 1
        con.commit()
        d += timedelta(days=1)
    if verbose:
        print(f"  espn: {n} game-rows through {end}")


def build_features(con, completed_only: bool = True) -> pd.DataFrame:
    q = ("SELECT * FROM nba_games WHERE game_type='REG' "
         "ORDER BY date, game_id")
    g = pd.read_sql(q, con)
    g["dt"] = pd.to_datetime(g.date)
    rows = []
    for season, sg in g.groupby("season", sort=True):
        pf = defaultdict(float); pa = defaultdict(float)
        gp = defaultdict(int); last: dict = {}

        def pyth(t):
            f = pf[t] + PRIOR_PPG * PRIOR_G
            a = pa[t] + PRIOR_PPG * PRIOR_G
            return f ** PYTH_EXP / (f ** PYTH_EXP + a ** PYTH_EXP)

        for r in sg.itertuples():
            h, a = r.home_id, r.away_id
            done = bool(r.completed)
            if gp[h] >= BURN_IN and gp[a] >= BURN_IN \
                    and (done or not completed_only):
                ra = (r.dt - last[a]).days if a in last else REST_CAP
                rh = (r.dt - last[h]).days if h in last else REST_CAP
                rows.append({
                    "game_id": r.game_id, "season": season,
                    "date": r.date,
                    "away_abbr": r.away_abbr, "home_abbr": r.home_abbr,
                    "away_score": r.away_score, "home_score": r.home_score,
                    "pyth_diff": pyth(a) - pyth(h),
                    "b2b_away": int(ra == 1), "b2b_home": int(rh == 1),
                    "rest_diff": min(REST_CAP, ra) - min(REST_CAP, rh),
                    "away_win": int((r.away_score or 0) > (r.home_score or 0)),
                    "completed": int(done),
                })
            if done:
                pf[h] += r.home_score; pa[h] += r.away_score; gp[h] += 1
                pf[a] += r.away_score; pa[a] += r.home_score; gp[a] += 1
            last[h] = r.dt; last[a] = r.dt
    df = pd.DataFrame(rows)
    return df[df.completed == 1] if completed_only and len(df) else df


# ---------------------------------------------------------------- adapter
from types import SimpleNamespace


def _not_wired(*_a, **_k):
    raise NotImplementedError(
        "NBA odds/ledger wiring lands with the season-launch session; "
        "the model core, ingest, and features are validated and ready.")


def ingest(con, season: int) -> None:
    """Season ingest: bootstrap history once, then top up via ESPN."""
    have = con.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE name='nba_games'"
    ).fetchone()[0]
    if not have or not con.execute(
            "SELECT COUNT(*) FROM nba_games").fetchone()[0]:
        ingest_master(con)
    if season >= 2024:
        ingest_espn_range(con, date(season - 1, 10, 1),
                          min(date.today(), date(season, 6, 30)), season)


ADAPTER = SimpleNamespace(
    key="nba",
    name="NBA",
    odds_sport="basketball_nba",
    books=["draftkings", "fanduel", "hardrockbet"],
    data_prefix="nba/",
    team_name_to_id=None,          # wired with odds capture at launch
    season_bounds=lambda season: (f"{season-1}-10-01", f"{season}-06-30"),
    ingest=ingest,
    build_features=build_features,
    GROUPS=GROUPS,
    label=label,
    slate=_not_wired,
    slate_features=_not_wired,
    baseline=None,
)
