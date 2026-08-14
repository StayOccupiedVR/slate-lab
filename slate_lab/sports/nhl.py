"""NHL adapter. Validated 2026-08-13 on fastRhockey history (2010-2023).

VALIDATION (walk-forward logistic, 14,678 feature-games; holdouts
trained on all prior seasons):
    2021: acc .589  logloss .6689  (baseline .6941)
    2022: acc .615  logloss .6567  (baseline .6907)
    2023: acc .600  logloss .6626  (baseline .6934)
Hockey is the highest-variance major sport — closing lines themselves
sit near .655 log loss — so this is the honest public-data band. The
schedule effects carry real weight and the right signs: a HOME team on
a back-to-back is the biggest penalty in the sport (+0.226 toward the
visitor), road b2b -0.132, small rest edge.

FEATURES (point-in-time, 8-game burn-in per team per season):
    pyth_diff   Pythagorean exponent 2.15 (hockey's classic),
                prior 8 games at 3.0 goals/game
    b2b_away / b2b_home   second night of a back-to-back (~28% of
                games — dense schedule, big lever)
    rest_diff   capped rest-day differential

DATA TRAP (cost a full debugging pass — do not repeat): the master's
`date` column is NULL for every REG/Final row; the populated column
is `game_date`. A naive `date` sort produces NaT ordering and silently
kills every schedule feature. Always build from `game_date`.

DATA    History bootstrap: sportsdataverse fastRhockey master parquet
        on GitHub (2009-2023). Current seasons: ESPN's public NHL
        scoreboard API from Actions, one call per date.
"""
from __future__ import annotations

import io
import json
import urllib.request
from collections import defaultdict
from datetime import date, timedelta

import numpy as np
import pandas as pd

PYTH_EXP = 2.15
PRIOR_G = 8
PRIOR_GPG = 3.0
BURN_IN = 8
REST_CAP = 3

MASTER_URL = ("https://raw.githubusercontent.com/sportsdataverse/"
              "fastRhockey-data/main/nhl_schedule_master.parquet")
ESPN_URL = ("https://site.api.espn.com/apis/site/v2/sports/hockey/"
            "nhl/scoreboard?dates={ymd}")

SCHEMA = """
CREATE TABLE IF NOT EXISTS nhl_games (
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
CREATE INDEX IF NOT EXISTS idx_nhl_season ON nhl_games(season, date);
"""

label = "away_win"
GROUPS = {
    "team": ["pyth_diff"],
    "schedule": ["b2b_away", "b2b_home", "rest_diff"],
}
EVAL_COLS = ["game_id", "season", "date", "away_abbr", "home_abbr",
             "away_score", "home_score"]


def ingest_master(con) -> None:
    """Bootstrap 2009-2023 history. Uses game_date (see DATA TRAP)."""
    con.executescript(SCHEMA)
    raw = urllib.request.urlopen(MASTER_URL, timeout=180).read()
    df = pd.read_parquet(io.BytesIO(raw))
    df = df[(df.status_detailed_state == "Final")
            & df.home_score.notna() & df.away_score.notna()
            & df.game_date.notna()]
    rows = []
    for r in df.itertuples():
        rows.append((str(r.game_id), int(r.season),
                     str(pd.to_datetime(r.game_date).date()),
                     str(r.away_team_id), str(r.home_team_id),
                     getattr(r, "away_team_name", None),
                     getattr(r, "home_team_name", None),
                     float(r.away_score), float(r.home_score), 1,
                     "REG" if r.game_type == "REG" else "OTHER"))
    con.executemany(
        "INSERT OR REPLACE INTO nhl_games VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        rows)
    con.commit()
    print(f"  master: {len(rows)} games")


def ingest_espn_range(con, start: date, end: date,
                      season: int, verbose: bool = True) -> None:
    """Current-season top-up from ESPN, one polite call per day."""
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
                "INSERT OR REPLACE INTO nhl_games VALUES "
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
    q = ("SELECT * FROM nhl_games WHERE game_type='REG' "
         "ORDER BY date, game_id")
    g = pd.read_sql(q, con)
    g["dt"] = pd.to_datetime(g.date)
    rows = []
    for season, sg in g.groupby("season", sort=True):
        pf = defaultdict(float); pa = defaultdict(float)
        gp = defaultdict(int); last: dict = {}

        def pyth(t):
            f = pf[t] + PRIOR_GPG * PRIOR_G
            a = pa[t] + PRIOR_GPG * PRIOR_G
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
        "NHL odds/ledger wiring lands with the season-launch session; "
        "the model core, ingest, and features are validated and ready.")


def ingest(con, season: int) -> None:
    have = con.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE name='nhl_games'"
    ).fetchone()[0]
    if not have or not con.execute(
            "SELECT COUNT(*) FROM nhl_games").fetchone()[0]:
        ingest_master(con)
    if season >= 2024:
        ingest_espn_range(con, date(season - 1, 10, 1),
                          min(date.today(), date(season, 6, 30)), season)


ADAPTER = SimpleNamespace(
    key="nhl",
    name="NHL",
    odds_sport="icehockey_nhl",
    books=["draftkings", "fanduel", "hardrockbet"],
    data_prefix="nhl/",
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
