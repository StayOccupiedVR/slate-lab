"""Regenerates nfl-backtest.json — the public NFL receipts page data.

Run when models change; the output is a static artifact committed to
the slate-app repo. Requires network (nflverse) and ~5 minutes.

    python experiments/nfl_backtest_report.py --out nfl-backtest.json

Walks: moneyline logistic + margin regression vs closing lines
(2015+ features, 2023/24/25 holdouts); all continuous prop markets'
quantile coverage; receptions NB vs Poisson. Numbers must match the
findings blocks in slate_lab/sports/nfl.py and slate_lab/nfl_props.py.
"""
# Implementation intentionally mirrors the validated session scripts;
# see repo history 2026-08-11/12. Kept as one file for reproducibility.
import argparse
import io
import json
import sqlite3
import sys
import urllib.request
from collections import defaultdict
from math import exp, factorial
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slate_lab.nfl_props import (MARKETS, MIN_GAMES, SCHEMA,  # noqa: E402
                                 count_distribution, fit_alpha,
                                 ingest_player_weeks, yards_distribution)


def games_walk():
    raw = urllib.request.urlopen(
        "https://github.com/nflverse/nfldata/raw/master/data/games.csv",
        timeout=120).read()
    df = pd.read_csv(io.BytesIO(raw))
    g = df[(df.season >= 2015) & (df.game_type == "REG")
           & df.result.notna()].sort_values(["season", "week"])
    rows = []
    for season, sg in g.groupby("season", sort=True):
        pf = defaultdict(float); pa = defaultdict(float)
        gp = defaultdict(int); qb = {}
        def pyth(t):
            f, a = pf[t] + 21.0 * 6, pa[t] + 21.0 * 6
            return f ** 2.37 / (f ** 2.37 + a ** 2.37)
        for r in sg.itertuples():
            if gp[r.away_team] >= 3 and gp[r.home_team] >= 3:
                imp = None
                if pd.notna(r.away_moneyline) and pd.notna(r.home_moneyline):
                    a_ = (100 / (r.away_moneyline + 100) if r.away_moneyline > 0
                          else -r.away_moneyline / (-r.away_moneyline + 100))
                    h_ = (100 / (r.home_moneyline + 100) if r.home_moneyline > 0
                          else -r.home_moneyline / (-r.home_moneyline + 100))
                    imp = a_ / (a_ + h_)
                rows.append(dict(
                    season=season,
                    pyth_diff=pyth(r.away_team) - pyth(r.home_team),
                    rest_diff=((r.away_rest - r.home_rest)
                               if pd.notna(r.away_rest) else 0),
                    qb_new_away=int(qb.get(r.away_team) is not None
                                    and qb.get(r.away_team) != r.away_qb_name),
                    qb_new_home=int(qb.get(r.home_team) is not None
                                    and qb.get(r.home_team) != r.home_qb_name),
                    div=int(r.div_game) if pd.notna(r.div_game) else 0,
                    away_win=int(r.result < 0),
                    home_margin=float(r.result),
                    spread=(float(r.spread_line)
                            if pd.notna(r.spread_line) else None),
                    mkt_p_away=imp))
            pf[r.away_team] += r.away_score; pa[r.away_team] += r.home_score
            gp[r.away_team] += 1
            pf[r.home_team] += r.home_score; pa[r.home_team] += r.away_score
            gp[r.home_team] += 1
            qb[r.away_team] = r.away_qb_name; qb[r.home_team] = r.home_qb_name
    F = pd.DataFrame(rows)
    feats = ["pyth_diff", "rest_diff", "qb_new_away", "qb_new_home", "div"]
    ml, mg = [], []
    for season in (2023, 2024, 2025):
        tr, te = F[F.season < season], F[F.season == season]
        m = LogisticRegression(C=1.0).fit(tr[feats], tr.away_win)
        p = m.predict_proba(te[feats])[:, 1]
        mm = LinearRegression().fit(tr[feats], tr.home_margin)
        mkt = te[te.mkt_p_away.notna()]
        ml.append({"season": season, "n": int(len(te)),
                   "acc": round(float(((p >= .5) == (te.away_win == 1)).mean()), 3),
                   "logloss": round(float(log_loss(te.away_win, p)), 4),
                   "market_logloss": round(float(log_loss(
                       mkt.away_win, mkt.mkt_p_away)), 4),
                   "market_n": int(len(mkt))})
        sp = te[te.spread.notna()]
        mg.append({"season": season,
                   "margin_mae": round(float(np.abs(
                       mm.predict(te[feats]) - te.home_margin).mean()), 2),
                   "market_margin_mae": round(float(np.abs(
                       sp.spread - sp.home_margin).mean()), 2)})
    return ml, mg


def props_walk(con):
    ps = pd.read_sql("SELECT * FROM nfl_player_weeks ORDER BY season, week", con)
    QS = (0.10, 0.25, 0.50, 0.75, 0.90)
    v = lambda x: x if x is not None else 0
    cont = {}
    for mk, (poss, elig, val, kind, w) in MARKETS.items():
        if kind != "cont":
            continue
        cont[mk] = []
        for season in (2023, 2024, 2025):
            hist = defaultdict(list); base = []
            for r in ps[ps.season < season].itertuples():
                if r.position in poss.split() and elig(r) and val(r) is not None:
                    hist[r.player_id].append(val(r)); base.append(val(r))
            cov = {q: 0 for q in QS}; n = 0
            h2 = {k: list(x) for k, x in hist.items()}; b2 = list(base)
            for r in ps[ps.season == season].itertuples():
                if r.position in poss.split() and elig(r) and val(r) is not None:
                    d = yards_distribution(h2.get(r.player_id, []),
                                           np.array(b2), win=w)
                    if d:
                        n += 1
                        for q in QS:
                            if val(r) <= d["q"][str(int(q * 100))]:
                                cov[q] += 1
                    h2.setdefault(r.player_id, []).append(val(r))
                    b2.append(val(r))
            cont[mk].append({"season": season, "n": n,
                             **{f"q{int(q*100)}": round(100 * cov[q] / n, 1)
                                for q in QS}})
    rec = []
    for season in (2023, 2024, 2025):
        hist = defaultdict(list); pairs = []
        for r in ps[ps.season < season].itertuples():
            if r.position in ("WR", "TE", "RB") and v(r.targets) > 0 \
                    and r.receptions is not None:
                pr = hist[r.player_id]
                if len(pr) >= MIN_GAMES:
                    pairs.append((float(np.mean(pr[-16:])), r.receptions))
                pr.append(r.receptions)
        alpha = fit_alpha(pairs)
        h2 = {k: list(x) for k, x in hist.items()}
        ll = []; lln = []; op = []; oa = []
        for r in ps[ps.season == season].itertuples():
            if r.position in ("WR", "TE", "RB") and v(r.targets) > 0 \
                    and r.receptions is not None:
                pr = h2.get(r.player_id, [])
                d = count_distribution(pr, alpha, max_k=12)
                if d is not None:
                    k = min(int(r.receptions), 12)
                    pmf = np.array(d["pmf"])
                    ll.append(-np.log(max(pmf[k], 1e-9)))
                    mu = max(.05, np.mean(pr[-16:]))
                    nv = np.array([exp(-mu) * mu**i / factorial(i)
                                   for i in range(13)])
                    nv[-1] += max(0, 1 - nv.sum())
                    lln.append(-np.log(max(nv[k], 1e-9)))
                    op.append(float(pmf[4:].sum()))
                    oa.append(1.0 if r.receptions > 3.5 else 0.0)
                h2.setdefault(r.player_id, []).append(r.receptions)
        pred = np.array(op); act = np.array(oa); bk = []
        for lo in (0, .2, .4, .6, .8):
            m = (pred >= lo) & (pred < lo + .2)
            if m.sum() >= 40:
                bk.append({"pred": round(100 * float(pred[m].mean())),
                           "hit": round(100 * float(act[m].mean())),
                           "n": int(m.sum())})
        rec.append({"season": season, "n": len(ll), "alpha": round(alpha, 3),
                    "nb_log": round(float(np.mean(ll)), 4),
                    "poisson_log": round(float(np.mean(lln)), 4),
                    "p4_buckets": bk})
    return cont, rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="nfl-backtest.json")
    args = p.parse_args()
    con = sqlite3.connect(":memory:")
    con.executescript(SCHEMA)
    for s in range(2020, 2026):
        ingest_player_weeks(con, s)
    ml, mg = games_walk()
    cont, rec = props_walk(con)
    doc = {
        "generated": pd.Timestamp.today().date().isoformat(),
        "model": "nfl-v1",
        "moneyline": ml, "margin": mg,
        "spread_note": ("Cover probabilities were tested and REJECTED: log "
                        "loss .71-.73 vs the .693 coin baseline on all three "
                        "holdouts; ATS 45-54% at every threshold. Slate "
                        "displays margins and grades their MAE publicly, and "
                        "does not sell cover picks."),
        "props_cont": cont, "receptions": rec,
        "benched": [
            {"market": "pass_tds", "why": "2025 calibration overconfident "
             "both directions (34->46, 67->56); inconsistent across seasons. "
             "Revisit as TD-rate-per-attempt."},
            {"market": "ints", "why": "NB ties/loses to Poisson; 2023 bucket "
             "33->55. ~0.5/game is too thin for a 16-game window."},
            {"market": "any_td", "why": "NB worse than Poisson on all three "
             "seasons and hot (48->38). Needs a red-zone opportunity model, "
             "planned as v2."},
        ],
    }
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=1)
    print(f"written -> {args.out}")


if __name__ == "__main__":
    main()
