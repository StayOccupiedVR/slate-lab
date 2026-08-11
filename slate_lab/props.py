"""Prop research v1 — MLB starter strikeout distributions.

WHAT     P(K = k) for a starting pitcher, built from data already in
         slate.db (the K-BB% ingest stores so/bb/bf per start).

MODEL    K rate = (career-to-date SO + prior) / (BF + prior), regressed
         over K_PRIOR_BF batters faced toward the league rate — K%
         stabilizes around 70 BF, the same result behind the K-BB%
         starter rating. Workload is not assumed but sampled: the
         pitcher's last WORKLOAD_N batters-faced values form an
         empirical mixture, and K | BF ~ Binomial(BF, rate). The
         mixture is the distribution the app displays and the
         probabilities P(over x.5) come straight off it.

HONEST   v1 knows the pitcher only. No opponent lineup K%, no park, no
LIMITS   umpire, no catcher. Those are v2 candidates that go through
         the same backtest gate as everything else. Displayed as
         research — no edge claims until lines exist to claim against.

BACKTEST python -m slate_lab.props --db slate.db --backtest 2025
         Walks every 2025 start point-in-time (only earlier starts
         feed each projection), scores the distribution's log score
         against a naive season-average Poisson, and prints
         calibration at the common lines (4.5-7.5).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from math import comb, exp, factorial

import numpy as np
import pandas as pd

K_PRIOR_BF = 70.0        # regress K rate over ~70 BF toward league
MIN_PRIOR_STARTS = 3     # don't project a pitcher with fewer priors
WORKLOAD_N = 200          # BF mixture support: effectively all prior starts. Synthetic ablation:
                         # last-10 lost to naive Poisson (2.1596 vs 2.1557), last-25 still behind
                         # (2.1572), full career won (2.1531). A recency window is a v2 candidate
                         # that must earn its place in the real-data backtest.
MAX_K = 16               # distribution support 0..MAX_K (16+ pooled)
LG_K_RATE_FALLBACK = 0.22
LG_BF_FALLBACK = 22


def _binom_pmf(n: int, p: float) -> np.ndarray:
    ks = np.arange(0, MAX_K + 1)
    pmf = np.array([comb(n, k) * p**k * (1 - p)**(n - k) if k <= n else 0.0
                    for k in ks])
    pmf[-1] += max(0.0, 1.0 - pmf.sum())      # pool the tail into MAX_K
    return pmf


def k_distribution(prior: list[tuple[int, int]], lg_rate: float) -> dict | None:
    """P(K=k) from prior (so, bf) starts. None if history is too thin."""
    prior = [(so, bf) for so, bf in prior if bf and bf > 0]
    if len(prior) < MIN_PRIOR_STARTS:
        return None
    so = sum(p[0] for p in prior)
    bf = sum(p[1] for p in prior)
    rate = (so + lg_rate * K_PRIOR_BF) / (bf + K_PRIOR_BF)
    workloads = [p[1] for p in prior[-WORKLOAD_N:]]
    pmf = np.zeros(MAX_K + 1)
    w = 1.0 / len(workloads)
    for n in workloads:
        pmf += w * _binom_pmf(int(n), rate)
    mean = float((np.arange(MAX_K + 1) * pmf).sum())
    return {
        "pmf": [round(float(x), 5) for x in pmf],
        "mean": round(mean, 2),
        "rate": round(rate, 4),
        "n_prior": len(prior),
        "over": {str(line): round(float(pmf[int(line) + 1:].sum()), 4)
                 for line in (3.5, 4.5, 5.5, 6.5, 7.5, 8.5)},
    }


def _poisson_pmf(lam: float) -> np.ndarray:
    ks = np.arange(0, MAX_K + 1)
    pmf = np.array([exp(-lam) * lam**k / factorial(k) for k in ks])
    pmf[-1] += max(0.0, 1.0 - pmf.sum())
    return pmf


def backtest(con, season: int) -> dict:
    sp = pd.read_sql(
        "SELECT pitcher_id, date, so, bf FROM pitcher_starts "
        "WHERE so IS NOT NULL AND bf IS NOT NULL AND bf > 0 "
        "ORDER BY date", con)
    lg_hist = sp[sp.date < f"{season}-01-01"]
    lg_rate = (float(lg_hist.so.sum()) / float(lg_hist.bf.sum())
               if len(lg_hist) else LG_K_RATE_FALLBACK)

    by_pid: dict = defaultdict(list)
    for r in sp[sp.date < f"{season}-01-01"].itertuples():
        by_pid[r.pitcher_id].append((r.so, r.bf))

    rows = []
    season_sp = sp[(sp.date >= f"{season}-01-01")
                   & (sp.date <= f"{season}-12-31")]
    for r in season_sp.itertuples():
        prior = by_pid.get(r.pitcher_id, [])
        d = k_distribution(prior, lg_rate)
        if d is not None:
            k = min(int(r.so), MAX_K)
            pmf = np.array(d["pmf"])
            naive_lam = max(0.5, sum(p[0] for p in prior)
                            / max(1, len(prior)))
            naive = _poisson_pmf(naive_lam)
            rows.append({
                "k": k,
                "ll": -np.log(max(pmf[k], 1e-9)),
                "ll_naive": -np.log(max(naive[k], 1e-9)),
                "over": {ln: p for ln, p in d["over"].items()},
            })
        by_pid[r.pitcher_id].append((r.so, r.bf))

    if not rows:
        return {"n": 0}
    ll = float(np.mean([r["ll"] for r in rows]))
    lln = float(np.mean([r["ll_naive"] for r in rows]))
    calib = {}
    for line in ("4.5", "5.5", "6.5", "7.5"):
        preds = np.array([float(r["over"][line]) for r in rows])
        actual = np.array([1.0 if r["k"] > float(line) else 0.0
                           for r in rows])
        buckets = []
        for lo in (0.0, 0.2, 0.4, 0.6, 0.8):
            m = (preds >= lo) & (preds < lo + 0.2)
            if m.sum() >= 25:
                buckets.append({"pred": round(float(preds[m].mean()), 3),
                                "hit": round(float(actual[m].mean()), 3),
                                "n": int(m.sum())})
        calib[line] = buckets
    return {"season": season, "n": len(rows),
            "logscore": round(ll, 4), "logscore_naive": round(lln, 4),
            "calibration": calib}


def print_backtest(res: dict) -> None:
    if not res.get("n"):
        print("No scoreable starts (need so/bf columns populated).")
        return
    print(f"season {res['season']}  starts scored: {res['n']}")
    print(f"  distribution log score {res['logscore']}   "
          f"naive Poisson {res['logscore_naive']}   "
          f"({'better' if res['logscore'] < res['logscore_naive'] else 'WORSE'})")
    for line, buckets in res["calibration"].items():
        parts = ", ".join(f"pred {b['pred']:.0%}->hit {b['hit']:.0%} "
                          f"(n={b['n']})" for b in buckets)
        print(f"  over {line}: {parts}")


def project_slate(con, slate: list[dict], date: str) -> dict:
    """Strikeout distributions for every probable starter on a slate.

    `slate` rows need away_sp/home_sp ids plus team abbreviations; pitcher
    names are looked up from the ids when the caller provides them in
    `names` (id -> name), else shown by id. Pure function of the db —
    no network — so it is testable and the workflow can call it right
    after scoring."""
    sp = pd.read_sql(
        "SELECT pitcher_id, date, so, bf FROM pitcher_starts "
        "WHERE so IS NOT NULL AND bf IS NOT NULL AND bf > 0 "
        "AND date < ? ORDER BY date", con, params=(date,))
    lg_rate = (float(sp.so.sum()) / float(sp.bf.sum())
               if len(sp) else LG_K_RATE_FALLBACK)
    hist = defaultdict(list)
    for r in sp.itertuples():
        hist[r.pitcher_id].append((r.so, r.bf))
    out = []
    for g in slate:
        for side, opp_side in (("away", "home"), ("home", "away")):
            pid = g.get(side + "_sp")
            if not pid:
                continue
            d = k_distribution(hist.get(pid, []), lg_rate)
            if d is None:
                continue
            out.append({
                "pitcher_id": pid,
                "name": (g.get(side + "_sp_name") or str(pid)),
                "team": g.get(side) or "",
                "opp": g.get(opp_side) or "",
                "gamePk": g.get("gamePk"),
                **d,
            })
    return {"date": date, "market": "strikeouts", "model": "props-k-v1",
            "pitchers": out}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="slate.db")
    p.add_argument("--backtest", type=int, default=None,
                   help="season to walk point-in-time")
    p.add_argument("--out", default=None,
                   help="write JSON here (backtest or projections)")
    p.add_argument("--project", default=None,
                   help="date YYYY-MM-DD: project today's probable starters")
    args = p.parse_args()
    con = sqlite3.connect(args.db)
    if args.backtest:
        res = backtest(con, args.backtest)
        print_backtest(res)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(res, f, indent=1)
            print(f"written -> {args.out}")
    if args.project:
        from .score import todays_slate
        raw = todays_slate(args.project)
        doc = project_slate(con, raw, args.project)
        out = args.out or "props-mlb.json"
        with open(out, "w") as f:
            json.dump(doc, f, indent=1)
        print(f"{len(doc['pitchers'])} starters projected -> {out}")


if __name__ == "__main__":
    main()
