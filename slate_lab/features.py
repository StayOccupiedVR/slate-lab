"""Point-in-time feature construction.

THE ONE RULE: every feature for a game on date D may use only rows with
date < D. tests/test_pipeline.py enforces this with a tampering tripwire —
if you add a feature here, extend that test or you have no proof it's clean.

Feature groups are registered in GROUPS so evaluate.py can run ablations:
each group must earn its place by improving held-out log loss.
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict

import pandas as pd

PYTH_EXP = 1.83
SP_PRIOR_IP = 50.0
SP_L5_WEIGHT = 0.25
IP_PER_START = 5.4

GROUPS: dict[str, list[str]] = {
    "team":     ["pyth_diff", "gp_min"],
    "kbb":      ["sp_kbb"],
}
# All measured WORSE than team+starter on the 2025 holdout (n=2123).
# Kept computed and re-testable, but out of the default model.
#   form / rest : raised log loss outright
#   rotation    : sp_vs_rot correlates -0.015 with outcome and -0.39 with
#                 pyth_diff -- a moving baseline (good staff makes every arm
#                 look bad) so it measures roster shape, not pitching.
LEGACY: dict[str, list[str]] = {
    # retired 2026-07-30: equal holdout logloss to kbb (0.6827 both) but ERA
    # stabilizes far slower and carries more team-strength collinearity
    # (corr w/ pyth_diff +0.31 vs +0.25); kbb also stronger w/ outcome
    # (+0.103 vs +0.081). Kept for re-testing.
    "starter":  ["sp_edge", "sp_known"],
    # retired 2026-07-30: +0.0001 in the ablation, below the noise floor
    # at n=2123. Real effect may exist; undetectable at this sample size.
    "park":     ["park_run"],
    "form":     ["l10_diff"],
    "rest":     ["rest_diff", "b2b_away", "b2b_home"],
    "rotation": ["sp_vs_rot"],
}
ROT_MIN_OUTS = 300          # ~100 IP before a team rotation baseline is trusted
KBB_PRIOR_BF = 170.0        # ~40 IP of batters faced; K% stabilizes ~70, BB% ~170

# Static 3-year park run factors (1.00 = neutral). Applied as (factor - 1) for
# the HOME park. Rationale for including at all: higher run environments add
# outcome variance, which slightly compresses favorite win probability. The
# ablation judges whether that effect is detectable at this sample size.
PARK_RUN = {
    115: 1.33, 111: 1.07, 113: 1.10, 118: 1.05, 109: 1.05, 133: 1.05,
    144: 1.04, 143: 1.04, 147: 1.02, 145: 1.01, 112: 1.01, 110: 1.00,
    142: 1.00, 120: 1.00, 158: 1.00, 108: 0.99, 141: 0.99, 117: 0.99,
    140: 0.98, 138: 0.97, 116: 0.96, 119: 0.96, 139: 0.96, 114: 0.95,
    146: 0.94, 134: 0.94, 135: 0.94, 121: 0.94, 136: 0.90, 137: 0.90,
}
LABEL = "away_won"


def build_features(con: sqlite3.Connection, min_gp: int = 20) -> pd.DataFrame:
    """One pass over the season in date order, accumulating state as we go.

    Accumulate-then-score ordering makes lookahead structurally impossible:
    a game's features are computed BEFORE its result is folded into state.
    """
    games = pd.read_sql(
        "SELECT * FROM games ORDER BY season, date, game_pk", con
    )
    sp = pd.read_sql(
        "SELECT * FROM pitcher_starts ORDER BY pitcher_id, date", con
    )
    sp_by_pid: dict[int, list[tuple[str, int, int]]] = defaultdict(list)
    kbb_by_pid: dict[int, list[tuple[str, float, float, float]]] = defaultdict(list)
    start_line: dict[tuple[int, str], tuple[int, int]] = {}
    for r in sp.itertuples():
        sp_by_pid[r.pitcher_id].append((r.date, r.er, r.outs))
        start_line[(r.pitcher_id, r.date)] = (r.er, r.outs)
        so = getattr(r, "so", None)
        bf = getattr(r, "bf", None)
        if so is not None and bf and bf > 0:
            kbb_by_pid[r.pitcher_id].append(
                (r.date, float(so), float(getattr(r, "bb", 0) or 0), float(bf)))

    rows = []
    for season, sgames in games.groupby("season", sort=True):
        team = defaultdict(lambda: {"w": 0, "l": 0, "rs": 0, "ra": 0,
                                    "recent": [], "last_date": None})
        rotation = defaultdict(lambda: {"er": 0, "outs": 0})
        lg_runs, lg_team_games = 0, 0
        lg_so, lg_bb, lg_bf = 0.0, 0.0, 0.0

        for g in sgames.itertuples():
            a, h = team[g.away_id], team[g.home_id]
            gp_a, gp_h = a["w"] + a["l"], h["w"] + h["l"]

            if gp_a >= min_gp and gp_h >= min_gp:
                lg_era = (lg_runs / max(lg_team_games, 1)) * 0.92
                pyth = lambda t: (t["rs"] ** PYTH_EXP) / (
                    t["rs"] ** PYTH_EXP + t["ra"] ** PYTH_EXP)
                l10 = lambda t: (sum(t["recent"][-10:]) / len(t["recent"][-10:])
                                 if t["recent"] else 0.5)
                rest = lambda t: (min(
                    (pd.Timestamp(g.date) - pd.Timestamp(t["last_date"])).days, 5)
                    if t["last_date"] else 3)

                def rot_era(tid):
                    r = rotation[tid]
                    return (r["er"] * 27 / r["outs"]
                            if r["outs"] >= ROT_MIN_OUTS else lg_era)

                era_a = _sp_era(sp_by_pid.get(g.away_sp), g.date, lg_era)
                era_h = _sp_era(sp_by_pid.get(g.home_sp), g.date, lg_era)
                ea = None if era_a is None else (lg_era - era_a) * IP_PER_START / 9
                eh = None if era_h is None else (lg_era - era_h) * IP_PER_START / 9
                ra_rot = (0.0 if era_a is None
                          else (rot_era(g.away_id) - era_a) * IP_PER_START / 9)
                rh_rot = (0.0 if era_h is None
                          else (rot_era(g.home_id) - era_h) * IP_PER_START / 9)

                lg_kbb = ((lg_so - lg_bb) / lg_bf) if lg_bf > 500 else 0.145
                ka = _sp_kbb(kbb_by_pid.get(g.away_sp), g.date, lg_kbb)
                kh = _sp_kbb(kbb_by_pid.get(g.home_sp), g.date, lg_kbb)

                rows.append({
                    "game_pk": g.game_pk, "date": g.date, "season": season,
                    "pyth_diff": pyth(a) - pyth(h),
                    "gp_min": min(gp_a, gp_h),
                    "l10_diff": l10(a) - l10(h),
                    "rest_diff": rest(a) - rest(h),
                    "b2b_away": int(rest(a) <= 1),
                    "b2b_home": int(rest(h) <= 1),
                    "sp_edge": (ea or 0.0) - (eh or 0.0),
                    "sp_known": int(ea is not None and eh is not None),
                    "sp_vs_rot": ra_rot - rh_rot,
                    "sp_kbb": ((ka if ka is not None else lg_kbb)
                               - (kh if kh is not None else lg_kbb)),
                    "park_run": PARK_RUN.get(g.home_id, 1.0) - 1.0,
                    LABEL: int(g.away_score > g.home_score),
                })

            # fold this game's result into state ONLY AFTER scoring it
            away_won = g.away_score > g.home_score
            for t, rs, ra, won in ((a, g.away_score, g.home_score, away_won),
                                   (h, g.home_score, g.away_score, not away_won)):
                t["rs"] += rs
                t["ra"] += ra
                t["w" if won else "l"] += 1
                t["recent"].append(int(won))
                t["last_date"] = g.date
                lg_runs += rs
                lg_team_games += 1

            for pid, tid in ((g.away_sp, g.away_id), (g.home_sp, g.home_id)):
                line = start_line.get((pid, g.date)) if pid else None
                if line:
                    rotation[tid]["er"] += line[0]
                    rotation[tid]["outs"] += line[1]
                for d, so, bb, bf in kbb_by_pid.get(pid, ()):
                    if d == g.date:
                        lg_so += so
                        lg_bb += bb
                        lg_bf += bf

    return pd.DataFrame(rows)


def _sp_era(log, date: str, lg_era: float) -> float | None:
    """Prior-regressed ERA as of `date`, blending season with last five."""
    if not log:
        return None
    before = [(er, outs) for d, er, outs in log if d < date]
    if not before or sum(o for _, o in before) < 9:
        return None
    er = sum(e for e, _ in before)
    outs = sum(o for _, o in before)
    era, ip = er * 27 / outs, outs / 3
    l5 = before[-5:]
    o5 = sum(o for _, o in l5)
    if o5 > 0:
        era = era * (1 - SP_L5_WEIGHT) + (sum(e for e, _ in l5) * 27 / o5) * SP_L5_WEIGHT
    return (era * ip + lg_era * SP_PRIOR_IP) / (ip + SP_PRIOR_IP)


def _sp_kbb(log, date: str, lg_kbb: float) -> float | None:
    """Prior-regressed K-BB% as of `date`. Returns None when unknown.

    K-BB% is used instead of ERA because it stabilizes far faster (K% in ~70
    batters faced, BB% in ~170) and excludes defense and sequencing luck.
    Regressed toward the league rate over KBB_PRIOR_BF batters faced."""
    if not log:
        return None
    before = [(so, bb, bf) for d, so, bb, bf in log if d < date]
    if not before:
        return None
    so = sum(x[0] for x in before)
    bb = sum(x[1] for x in before)
    bf = sum(x[2] for x in before)
    if bf < 30:
        return None
    raw = (so - bb) / bf
    return (raw * bf + lg_kbb * KBB_PRIOR_BF) / (bf + KBB_PRIOR_BF)


def feature_columns(groups: list[str] | None = None) -> list[str]:
    groups = groups or list(GROUPS)
    lookup = {**GROUPS, **LEGACY}
    return [c for grp in groups for c in lookup[grp]]
