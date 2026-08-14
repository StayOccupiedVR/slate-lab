"""NFL player props — full DK/Hard Rock-style menu. Validated 2026-08-12.

SHIPS (8 markets), walk-forward on 2023/24/25 holdouts:
  Continuous (prior-blended empirical pools; player window YD_WIN=24,
  QB markets QB_WIN=16 after a window ablation fixed a passing-decline
  drift; position baseline worth YARD_PRIOR_GAMES=12):
    rec_yards (WR/TE/RB)   coverage within ~3pts, q25 ~+2-3 (doc'd)
    rush_yards (RB 5+ car) within ~3pts
    pass_yards (QB 10+ att) q50 50-54 across seasons
    completions (QB)       q50 51-56 (was 52-60 at win=24 — fixed)
    pass_att (QB)          q50 53-56 stable
    rush_att (RB)          q50 52-55
    rush_rec_yds (RB)      q50 49-55
  Counts (negative binomial, dispersion fit by method of moments on
  training one-step pairs):
    receptions (WR/TE/RB, alpha~0.067): NB log 1.90/1.94/1.90 beats
    Poisson 1.91/1.95/1.90 all seasons; tails clean (10->10, 85->86);
    mid-range ~5pts hot (49->44) — documented, same precedent as the
    MLB K model's shipped high-confidence bias.

BENCHED (evidence, three seasons each):
  pass_tds  alpha=0 (Poisson-adequate variance) but 2025 overconfident
            both directions (34->46, 67->56); inconsistent across
            seasons. Revisit as TD-rate-per-attempt.
  ints      NB ties/loses to Poisson; calibration bad (33->55 in
            2023). Rate ~0.5/gm is too thin for a 16-game window.
  any_td    NB WORSE than Poisson all seasons and hot (48->38).
            Needs an opportunity model (red-zone touches), not a
            count window. Highest-demand bench — build it right.

LESSON  MLB workloads stable -> full career won; NFL roles volatile ->
        recency won, and QB volume needed the shortest window of all.

DATA    nflverse weekly player stats parquet, 2020+. Asset renamed
        mid-2025 (player_stats -> stats_player_week; interceptions ->
        passing_interceptions). Both handled.
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
QB_WIN = 16              # QB volume markets drift with league trends; shorter
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
  completions REAL,
  attempts  REAL,
  pass_yards REAL,
  pass_tds  REAL,
  ints      REAL,
  rush_tds  REAL,
  rec_tds   REAL,
  headshot  TEXT,
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
    int_col = ("interceptions" if "interceptions" in df.columns
               else "passing_interceptions")
    rows = [(r.player_id, int(r.season), int(r.week),
             getattr(r, "player_display_name", None), r.position,
             getattr(r, team_col, None),
             r.targets, r.receptions, r.receiving_yards,
             r.carries, r.rushing_yards,
             getattr(r, "completions", None), getattr(r, "attempts", None),
             getattr(r, "passing_yards", None),
             getattr(r, "passing_tds", None), getattr(r, int_col, None),
             getattr(r, "rushing_tds", None), getattr(r, "receiving_tds", None),
             getattr(r, "headshot_url", None))
            for r in df.itertuples()]
    con.executemany(
        "INSERT OR REPLACE INTO nfl_player_weeks VALUES "
        "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    con.commit()
    if verbose:
        print(f"  {season}: {len(rows)} player-weeks stored")


def yards_distribution(vals: list[float], pos_vals: np.ndarray,
                       rng=None, win: int | None = None) -> dict | None:
    """Prior-blended empirical yardage distribution.

    vals = the player's per-game yards (chronological); pos_vals = the
    position baseline pool. Returns quantiles and a callable-free pool
    summary the feed can serialize.
    """
    vals = [v for v in vals if v is not None][-(win or YD_WIN):]
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


COUNT_WIN = 16           # recency window for count markets


def _nb_pmf(mu: float, alpha: float, max_k: int) -> np.ndarray:
    """Negative binomial pmf, mean mu, Var = mu + alpha*mu^2.

    alpha -> 0 recovers Poisson; alpha is fit per market on training
    data by method of moments (pooled residual variance).
    """
    from math import lgamma, exp, log
    mu = max(mu, 1e-3)
    if alpha <= 1e-6:
        from math import factorial
        pmf = np.array([exp(-mu) * mu**k / factorial(k)
                        for k in range(max_k + 1)])
    else:
        r = 1.0 / alpha
        p = r / (r + mu)
        pmf = np.zeros(max_k + 1)
        for k in range(max_k + 1):
            pmf[k] = exp(lgamma(k + r) - lgamma(r) - lgamma(k + 1)
                         + r * log(p) + k * log(1 - p))
    pmf = np.clip(pmf, 0, None)
    pmf[-1] += max(0.0, 1.0 - pmf.sum())
    return pmf


def fit_alpha(pairs: list[tuple[float, float]]) -> float:
    """Method-of-moments dispersion from (predicted mu, actual) pairs:
    Var(y) = mu + alpha*mu^2 -> alpha = mean((y-mu)^2 - mu)/mean(mu^2)."""
    if not pairs:
        return 0.0
    mus = np.array([p[0] for p in pairs])
    ys = np.array([p[1] for p in pairs])
    num = float(np.mean((ys - mus) ** 2 - mus))
    den = float(np.mean(mus ** 2))
    return max(0.0, num / den) if den > 0 else 0.0


def count_distribution(vals: list[float], alpha: float,
                       max_k: int = 15) -> dict | None:
    """NB distribution for a count market from recent per-game counts."""
    vals = [v for v in vals if v is not None][-COUNT_WIN:]
    if len(vals) < MIN_GAMES:
        return None
    mu = float(np.mean(vals))
    pmf = _nb_pmf(mu, alpha, max_k)
    return {"mean": round(mu, 2), "n_prior": len(vals),
            "pmf": [round(float(x), 4) for x in pmf],
            "over": {f"{k}.5": round(float(pmf[k + 1:].sum()), 3)
                     for k in range(min(6, max_k))}}


# market registry: id -> (positions, eligibility col check, extractor, kind)
def _v(x):
    return x if x is not None else 0


MARKETS = {
    "rec_yards":   ("WR TE RB", lambda r: _v(r.targets) > 0,
                    lambda r: r.rec_yards, "cont", None),
    "rush_yards":  ("RB", lambda r: _v(r.carries) >= 5,
                    lambda r: r.rush_yards, "cont", None),
    "rush_rec_yds": ("RB", lambda r: _v(r.carries) >= 5 or _v(r.targets) >= 2,
                    lambda r: _v(r.rush_yards) + _v(r.rec_yards), "cont", None),
    "rush_att":    ("RB", lambda r: _v(r.carries) >= 5,
                    lambda r: r.carries, "cont", None),
    "pass_yards":  ("QB", lambda r: _v(r.attempts) >= 10,
                    lambda r: r.pass_yards, "cont", QB_WIN),
    "completions": ("QB", lambda r: _v(r.attempts) >= 10,
                    lambda r: r.completions, "cont", QB_WIN),
    "pass_att":    ("QB", lambda r: _v(r.attempts) >= 10,
                    lambda r: r.attempts, "cont", QB_WIN),
    "receptions":  ("WR TE RB", lambda r: _v(r.targets) > 0,
                    lambda r: r.receptions, "count", None),
}


def fit_receptions_alpha(con, before_season: int,
                         before_week: int | None = None) -> float:
    """One-step-ahead dispersion for the receptions NB, point-in-time."""
    q = "SELECT * FROM nfl_player_weeks WHERE season < ? ORDER BY season, week"
    df = pd.read_sql(q, con, params=(before_season,))
    if before_week is not None:
        cur = pd.read_sql(
            "SELECT * FROM nfl_player_weeks WHERE season = ? AND week < ? "
            "ORDER BY week", con, params=(before_season, before_week))
        df = pd.concat([df, cur])
    _, elig, val, _, _ = (
        MARKETS["receptions"][0], *MARKETS["receptions"][1:], )
    hist = defaultdict(list)
    pairs = []
    for r in df.itertuples():
        if r.position in ("WR", "TE", "RB") and _v(r.targets) > 0                 and r.receptions is not None:
            pr = hist[r.player_id]
            if len(pr) >= MIN_GAMES:
                pairs.append((float(np.mean(pr[-COUNT_WIN:])), r.receptions))
            pr.append(r.receptions)
    return fit_alpha(pairs)


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
    """Full prop menu for each game of a week. Players = top 8 by usage
    (targets+carries+attempts/2) over the last 5 played weeks per team.
    Every shipping market the player qualifies for is attached."""
    sched = pd.read_sql(
        "SELECT game_id, away_team, home_team, gameday FROM nfl_games "
        "WHERE season=? AND week=? AND game_type='REG' ORDER BY gameday",
        con, params=(season, week))
    q = "SELECT * FROM nfl_player_weeks WHERE season < ? ORDER BY season, week"
    hist_df = pd.read_sql(q, con, params=(season,))
    cur = pd.read_sql(
        "SELECT * FROM nfl_player_weeks WHERE season = ? AND week < ? "
        "ORDER BY week", con, params=(season, week))
    hist_df = pd.concat([hist_df, cur])
    hist = defaultdict(list)
    names = {}
    shots = {}
    base = defaultdict(list)     # (market, pos) -> values
    for r in hist_df.itertuples():
        hist[r.player_id].append(r)
        names[r.player_id] = r.name
        if getattr(r, "headshot", None):
            shots[r.player_id] = r.headshot
        for mk, (poss, elig, val, kind, _w) in MARKETS.items():
            if kind == "cont" and r.position in poss.split() and elig(r)                     and val(r) is not None:
                base[(mk, r.position)].append(val(r))
    rec_alpha = fit_receptions_alpha(con, season, week)
    recent = pd.read_sql(
        "SELECT player_id, team, position, "
        "SUM(COALESCE(targets,0)+COALESCE(carries,0)+COALESCE(attempts,0)/2) "
        "AS usage_ct FROM nfl_player_weeks "
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
                         "pos": pos, "team": team, "opp": opp,
                         "headshot": shots.get(pid)}
                for mk, (poss, elig, val, kind, w) in MARKETS.items():
                    if pos not in poss.split():
                        continue
                    vals = [val(x) for x in pr if elig(x)
                            and val(x) is not None]
                    if kind == "cont":
                        d = yards_distribution(
                            vals, np.array(base.get((mk, pos), [])), win=w)
                        if d:
                            entry[mk] = d
                    else:
                        d = count_distribution(vals, rec_alpha)
                        if d:
                            d = {k: v for k, v in d.items() if k != "pmf"}
                            entry[mk] = d
                if len(entry) > 5:
                    players.append(entry)
        games.append({"game_id": g.game_id, "away": g.away_team,
                      "home": g.home_team, "gameday": g.gameday,
                      "players": players})
    return {"season": season, "week": week, "market": "props-menu",
            "model": "nfl-props-v2", "rec_alpha": round(rec_alpha, 4),
            "games": games}


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
