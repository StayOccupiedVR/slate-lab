"""NFL adapter — real implementation, validated 2026-08-04.

DATA    nflverse games.csv fetched directly from GitHub (the nfl_data_py
        package is only a downloader; going direct avoids its build issues).
        Every historical game ships with the CLOSING SPREAD and BOTH
        MONEYLINES, so backtests compare against the actual market with no
        ledger required — 27 seasons of the comparison MLB had to earn.

MODEL   First validated pass (walk-forward, holdout log loss vs market):
            2022  model .638   market .596
            2023  model .649   market .621
            2024  model .603   market .556
        Roughly half the naive-to-market distance captured by four cheap
        feature families. EPA from play-by-play is the known headroom and
        the next build.

TEAM    Pythagorean on points, exponent 2.37, 6-game prior at 21 ppg.
QB      qb_new flag: today's starting QB differs from the team's last game.
        Knowable pre-kickoff (nflverse publishes starters). The EPA-rated
        QB feature replaces this later.
REST    rest-day differential — byes and short weeks are real here.
DIV     divisional-game flag: familiarity compresses spreads slightly.

VALIDATE  ~272 games/season; wide error bars. Two holdout seasons minimum
          before trusting a feature change. Market columns make every
          ablation a model-vs-Vegas statement.

LEDGER  Live picks start in September; grading vs DraftKings closing.

SPREAD FINDINGS (2026-08-11 session)
    Margin model (linear on the validated features): walk-forward MAE
    9.8-10.4 pts vs the closing spread's 9.3-9.9 — within half a point
    of Vegas, corr .78-.85 with the line. SHIPS for display + public
    margin-accuracy grading.
    Cover probabilities via empirical residual CDF: log loss .71-.73 vs
    the .693 coin baseline on all three holdouts; ATS picks 45-54%, no
    edge at any threshold. DOES NOT SHIP — deviations from the closing
    spread are anti-informative, so no cover picks, ever, until a model
    actually beats this bar. Key numbers confirmed (|margin| modes: 3
    then 7), so any future cover work stays on empirical distributions.

EPA FINDINGS (2026-08-04 session, full script: experiments/nfl_epa.py)
    Team net EPA/play (prior-blended): REJECTED. Correlates 0.971 with
    pyth_diff — point differential and cashed-in EPA are the same signal.
    Holdout deltas were within noise both directions.

    QB EPA/dropback (announced starter, prior 200 db, season decay 0.5):
    BENCHED, not shipped. Holdout log loss vs base:
        2023  .6487 -> .6454     2024  .6029 -> .5854     2025  .6324 -> .6489
    Helps two seasons materially, damages the most recent one in every
    formulation tried (raw / decayed / clipped +-0.10/.06/.03 — tighter
    clips shrink both the gain and the damage in proportion). Hyper-
    parameters were tuned while observing holdouts, so treat even the
    good cells as optimistic. Revisit in-season with live starter data;
    do not ship a feature that fails its most recent season into launch.
"""
from __future__ import annotations

import io
import urllib.request
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pandas as pd

GAMES_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"

PYTH_EXP = 2.37
PRIOR_GAMES = 6          # regress ratings over ~6 games ...
PRIOR_PPG = 21.0         # ... toward league-average scoring
MIN_GP = 3               # don't score a game until both teams played 3

# nflverse team codes are already clean 2-3 letter ids; we use them directly.
# The Odds API uses full names; this map feeds the ledger.
TEAM_NAME_TO_ID = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN",
    "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE",
    "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS nfl_games (
  game_id   TEXT PRIMARY KEY,
  season    INTEGER NOT NULL,
  week      INTEGER NOT NULL,
  gameday   TEXT,
  away_team TEXT NOT NULL,
  home_team TEXT NOT NULL,
  away_score REAL,
  home_score REAL,
  away_rest  REAL,
  home_rest  REAL,
  away_qb    TEXT,
  home_qb    TEXT,
  div_game   INTEGER,
  spread_line REAL,
  total_line  REAL,
  away_ml     REAL,
  home_ml     REAL,
  game_type   TEXT
);
CREATE INDEX IF NOT EXISTS idx_nfl_season ON nfl_games(season, week);
"""


def _fetch_games() -> pd.DataFrame:
    raw = urllib.request.urlopen(GAMES_URL, timeout=90).read()
    return pd.read_csv(io.BytesIO(raw))


def ingest(con, season: int) -> None:
    """Pull the nflverse games file and store the requested season.

    The file is one download for all seasons; per-season calls just filter.
    Idempotent via INSERT OR REPLACE on game_id.
    """
    con.executescript(SCHEMA)
    df = _fetch_games()
    df = df[df.season == season]
    if df.empty:
        print(f"  season {season}: nothing in nflverse yet")
        return
    rows = [(r.game_id, int(r.season), int(r.week), r.gameday,
             r.away_team, r.home_team,
             r.away_score, r.home_score,
             getattr(r, "away_rest", None), getattr(r, "home_rest", None),
             getattr(r, "away_qb_name", None), getattr(r, "home_qb_name", None),
             int(r.div_game) if pd.notna(getattr(r, "div_game", None)) else 0,
             getattr(r, "spread_line", None), getattr(r, "total_line", None),
             getattr(r, "away_moneyline", None), getattr(r, "home_moneyline", None),
             r.game_type)
            for r in df.itertuples()]
    con.executemany(
        "INSERT OR REPLACE INTO nfl_games VALUES "
        "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    con.commit()
    played = df.away_score.notna().sum()
    print(f"  season {season}: {len(df)} games stored ({played} played)")


def _ml_to_prob(ml) -> float | None:
    if ml is None or (isinstance(ml, float) and np.isnan(ml)):
        return None
    ml = float(ml)
    return (-ml) / ((-ml) + 100) if ml < 0 else 100 / (ml + 100)


def build_features(con) -> pd.DataFrame:
    """Point-in-time features for every scoreable regular-season game.

    Ratings fold in strictly after each game is scored — the same
    walk-the-season pattern the MLB tripwire verified. Market columns
    (devigged closing probability, closing spread) ride along for
    evaluation only; they are never model inputs.
    """
    g = pd.read_sql(
        "SELECT * FROM nfl_games WHERE game_type='REG' "
        "AND away_score IS NOT NULL ORDER BY season, week, gameday", con)
    rows = []
    for season, sg in g.groupby("season", sort=True):
        pf: dict = defaultdict(float)
        pa: dict = defaultdict(float)
        gp: dict = defaultdict(int)
        qb_last: dict = {}

        def pyth(t):
            f = pf[t] + PRIOR_PPG * PRIOR_GAMES
            a = pa[t] + PRIOR_PPG * PRIOR_GAMES
            return f ** PYTH_EXP / (f ** PYTH_EXP + a ** PYTH_EXP)

        for r in sg.itertuples():
            if gp[r.away_team] >= MIN_GP and gp[r.home_team] >= MIN_GP:
                mp_a = _ml_to_prob(r.away_ml)
                mp_h = _ml_to_prob(r.home_ml)
                market = (mp_a / (mp_a + mp_h)
                          if mp_a is not None and mp_h is not None else None)
                qb_new_a = int(qb_last.get(r.away_team) is not None
                               and qb_last.get(r.away_team) != r.away_qb)
                qb_new_h = int(qb_last.get(r.home_team) is not None
                               and qb_last.get(r.home_team) != r.home_qb)
                rest = 0.0
                if r.away_rest is not None and r.home_rest is not None \
                        and not (pd.isna(r.away_rest) or pd.isna(r.home_rest)):
                    rest = float(r.away_rest - r.home_rest)
                rows.append({
                    "game_id": r.game_id, "season": int(season),
                    "week": int(r.week), "date": r.gameday,
                    "pyth_diff": pyth(r.away_team) - pyth(r.home_team),
                    "rest_diff": rest,
                    "qb_new_away": qb_new_a, "qb_new_home": qb_new_h,
                    "div": int(r.div_game or 0),
                    "away_won": int(r.away_score > r.home_score),
                    "margin": float(r.away_score - r.home_score),
                    "market_p_away": market,
                    "close_spread": r.spread_line,
                })
            pf[r.away_team] += r.away_score
            pa[r.away_team] += r.home_score
            gp[r.away_team] += 1
            pf[r.home_team] += r.home_score
            pa[r.home_team] += r.away_score
            gp[r.home_team] += 1
            qb_last[r.away_team] = r.away_qb
            qb_last[r.home_team] = r.home_qb
    return pd.DataFrame(rows)


GROUPS: dict[str, list[str]] = {
    "team": ["pyth_diff"],
    "situ": ["rest_diff", "div"],
    "qb":   ["qb_new_away", "qb_new_home"],
}

# Evaluation-only columns; the trainer must never feed these to a model.
EVAL_COLS = ["market_p_away", "close_spread", "margin"]


def season_bounds(season: int):
    return (f"{season}-09-01", f"{season + 1}-02-28")


def _slate(date: str):
    raise SystemExit(
        "NFL live slates start with the September ledger. Backtests and "
        "ablations are fully operational now: ingest + train --sport nfl.")


def _slate_features(con, slate, date):
    _slate(date)


ADAPTER = SimpleNamespace(
    key="nfl",
    name="NFL",
    odds_sport="americanfootball_nfl",
    books=["draftkings", "fanduel", "hardrockbet"],
    data_prefix="nfl/",
    team_name_to_id=TEAM_NAME_TO_ID,
    season_bounds=season_bounds,
    ingest=ingest,
    build_features=build_features,
    GROUPS=GROUPS,
    label="away_won",
    slate=_slate,
    slate_features=_slate_features,
    baseline=None,
)
