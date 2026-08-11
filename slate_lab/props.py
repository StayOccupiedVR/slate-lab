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


# ---- batting: hits and home runs share the binomial-mixture core ----
HIT_PRIOR_AB = 250       # BA stabilizes slowly (~900+ AB); heavy shrink by design
HR_PRIOR_AB = 300        # HR/AB stabilizes around ~170 PA; shrink a bit past it
MIN_PRIOR_GAMES = 30     # don't project a batter on fewer prior games
BAT_MAX = 6              # hits/HR support 0..6 (6+ pooled)
LG_HIT_RATE_FALLBACK = 0.245
LG_HR_RATE_FALLBACK = 0.032


def _bin_mix(pairs: list[tuple[int, int]], prior_n: float, lg_rate: float,
             max_k: int) -> dict | None:
    """Binomial mixture over empirical opportunity counts.

    pairs = (successes, opportunities) per prior game. Rate is
    prior-regressed; the opportunity mixture uses the full history —
    the same full-support finding the strikeout ablation produced.
    """
    pairs = [(a, b) for a, b in pairs if b and b > 0]
    if not pairs:
        return None
    succ = sum(p[0] for p in pairs)
    opp = sum(p[1] for p in pairs)
    rate = (succ + lg_rate * prior_n) / (opp + prior_n)
    pmf = np.zeros(max_k + 1)
    w = 1.0 / len(pairs)
    for _, n in pairs:
        n = min(int(n), 12)
        row = np.array([comb(n, k) * rate**k * (1 - rate)**(n - k)
                        if k <= n else 0.0 for k in range(max_k + 1)])
        row[-1] += max(0.0, 1.0 - row.sum())
        pmf += w * row
    mean = float((np.arange(max_k + 1) * pmf).sum())
    return {"pmf": pmf, "mean": mean, "rate": rate}


def hit_distribution(prior: list[tuple[int, int]], lg_rate: float) -> dict | None:
    """prior = (hits, at-bats) per game, oldest first."""
    if len(prior) < MIN_PRIOR_GAMES:
        return None
    d = _bin_mix(prior, HIT_PRIOR_AB, lg_rate, BAT_MAX)
    if d is None:
        return None
    pmf = d["pmf"]
    return {"mean": round(d["mean"], 2), "rate": round(d["rate"], 4),
            "n_prior": len(prior),
            "over": {"0.5": round(float(pmf[1:].sum()), 4),
                     "1.5": round(float(pmf[2:].sum()), 4),
                     "2.5": round(float(pmf[3:].sum()), 4)}}


def hr_distribution(prior: list[tuple[int, int]], lg_rate: float) -> dict | None:
    """prior = (home runs, at-bats) per game, oldest first."""
    if len(prior) < MIN_PRIOR_GAMES:
        return None
    d = _bin_mix(prior, HR_PRIOR_AB, lg_rate, BAT_MAX)
    if d is None:
        return None
    pmf = d["pmf"]
    return {"mean": round(d["mean"], 3), "rate": round(d["rate"], 4),
            "n_prior": len(prior),
            "over": {"0.5": round(float(pmf[1:].sum()), 4),
                     "1.5": round(float(pmf[2:].sum()), 4)}}


def backtest_batting(con, season: int) -> dict:
    bg = pd.read_sql(
        "SELECT batter_id, date, ab, h, hr FROM batter_games "
        "WHERE ab IS NOT NULL AND ab > 0 ORDER BY date", con)
    hist_lg = bg[bg.date < f"{season}-01-01"]
    lg_hit = (float(hist_lg.h.sum()) / float(hist_lg.ab.sum())
              if len(hist_lg) else LG_HIT_RATE_FALLBACK)
    lg_hr = (float(hist_lg.hr.sum()) / float(hist_lg.ab.sum())
             if len(hist_lg) else LG_HR_RATE_FALLBACK)
    hist: dict = defaultdict(list)
    for r in hist_lg.itertuples():
        hist[r.batter_id].append((r.h, r.hr, r.ab))
    rows = []
    for r in bg[(bg.date >= f"{season}-01-01")
                & (bg.date <= f"{season}-12-31")].itertuples():
        pr = hist.get(r.batter_id, [])
        if len(pr) >= MIN_PRIOR_GAMES:
            hd = hit_distribution([(h, ab) for h, _, ab in pr], lg_hit)
            rd = hr_distribution([(hr, ab) for _, hr, ab in pr], lg_hr)
            if hd and rd:
                rows.append({
                    "h": int(r.h), "hr": int(r.hr),
                    "p1": hd["over"]["0.5"], "p2": hd["over"]["1.5"],
                    "p3": hd["over"]["2.5"],
                    "phr": rd["over"]["0.5"], "phr2": rd["over"]["1.5"]})
        hist.setdefault(r.batter_id, []).append((r.h, r.hr, r.ab))
    if not rows:
        return {"n": 0}
    df = pd.DataFrame(rows)
    def calib(pred, actual):
        out = []
        for lo in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            m = (pred >= lo) & (pred < lo + 0.1)
            if m.sum() >= 50:
                out.append({"pred": round(float(pred[m].mean()), 3),
                            "hit": round(float(actual[m].mean()), 3),
                            "n": int(m.sum())})
        return out
    return {"season": season, "n": len(df),
            "calibration": {
                "hits_1plus": calib(df.p1, (df.h >= 1).astype(float)),
                "hits_2plus": calib(df.p2, (df.h >= 2).astype(float)),
                "hits_3plus": calib(df.p3, (df.h >= 3).astype(float)),
                "hr_1plus": calib(df.phr, (df.hr >= 1).astype(float)),
                "hr_2plus": calib(df.phr2, (df.hr >= 2).astype(float))}}


def print_batting(res: dict) -> None:
    if not res.get("n"):
        print("No scoreable batter games — run the batting ingest first.")
        return
    print(f"season {res['season']}  batter-games scored: {res['n']}")
    for mk, buckets in res["calibration"].items():
        parts = ", ".join(f"{b['pred']:.0%}->{b['hit']:.0%} (n={b['n']})"
                          for b in buckets)
        print(f"  {mk}: {parts or 'no full buckets'}")


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


def _team_offense_quality(con, date: str) -> dict:
    """team_id -> batting average, season-to-date of `date`'s season."""
    season = date[:4]
    df = pd.read_sql(
        "SELECT team_id, SUM(h) AS h, SUM(ab) AS ab FROM batter_games "
        "WHERE date >= ? AND date < ? AND ab > 0 GROUP BY team_id",
        con, params=(f"{season}-01-01", date))
    return {int(r.team_id): (r.h / r.ab) for r in df.itertuples()
            if r.ab and r.ab > 100}


def _team_pitching_quality(con, date: str) -> dict:
    """team_id -> runs allowed per game (lower = better staff)."""
    season = date[:4]
    df = pd.read_sql(
        "SELECT away_id, home_id, away_score, home_score FROM games "
        "WHERE date >= ? AND date < ?", con,
        params=(f"{season}-01-01", date))
    ra, gp = {}, {}
    for r in df.itertuples():
        ra[r.away_id] = ra.get(r.away_id, 0) + r.home_score
        ra[r.home_id] = ra.get(r.home_id, 0) + r.away_score
        gp[r.away_id] = gp.get(r.away_id, 0) + 1
        gp[r.home_id] = gp.get(r.home_id, 0) + 1
    return {t: ra[t] / gp[t] for t in ra if gp[t] >= 10}


def _split(pairs_with_opp, quality: dict, top_ids: set):
    """(succ, opp_count, opp_id) rows -> rate vs top-half and rest."""
    hi_s = hi_n = lo_s = lo_n = 0
    for s_, n_, oid in pairs_with_opp:
        if oid in top_ids:
            hi_s += s_; hi_n += n_
        elif oid in quality:
            lo_s += s_; lo_n += n_
    return {
        "hi": {"rate": round(hi_s / hi_n, 3) if hi_n else None, "n": hi_n},
        "lo": {"rate": round(lo_s / lo_n, 3) if lo_n else None, "n": lo_n},
    }


def project_slate(con, slate: list[dict], date: str) -> dict:
    """Strikeout distributions for every probable starter on a slate.

    `slate` rows need away_sp/home_sp ids plus team abbreviations; pitcher
    names are looked up from the ids when the caller provides them in
    `names` (id -> name), else shown by id. Pure function of the db —
    no network — so it is testable and the workflow can call it right
    after scoring."""
    sp = pd.read_sql(
        "SELECT pitcher_id, date, so, bf, opp_id FROM pitcher_starts "
        "WHERE so IS NOT NULL AND bf IS NOT NULL AND bf > 0 "
        "AND date < ? ORDER BY date", con, params=(date,))
    lg_rate = (float(sp.so.sum()) / float(sp.bf.sum())
               if len(sp) else LG_K_RATE_FALLBACK)
    hist = defaultdict(list)
    hist_full = defaultdict(list)
    for r in sp.itertuples():
        hist[r.pitcher_id].append((r.so, r.bf))
        hist_full[r.pitcher_id].append(
            (r.date, int(r.so), int(r.bf),
             int(r.opp_id) if pd.notna(r.opp_id) else None))
    off_q = _team_offense_quality(con, date)
    off_top = {t for t in off_q
               if off_q[t] >= sorted(off_q.values())[len(off_q)//2]} \
        if off_q else set()
    pitch_q = _team_pitching_quality(con, date)
    pitch_top = {t for t in pitch_q
                 if pitch_q[t] <= sorted(pitch_q.values())[len(pitch_q)//2]} \
        if pitch_q else set()
    out = []
    for g in slate:
        for side, opp_side in (("away", "home"), ("home", "away")):
            pid = g.get(side + "_sp")
            if not pid:
                continue
            d = k_distribution(hist.get(pid, []), lg_rate)
            if d is None:
                continue
            full = hist_full.get(pid, [])
            last10 = [{"date": dt, "so": so_, "bf": bf_}
                      for dt, so_, bf_, _ in full[-10:]][::-1]
            spl = _split([(so_, bf_, oid) for _, so_, bf_, oid in full],
                         off_q, off_top)
            out.append({
                "pitcher_id": pid,
                "name": (g.get(side + "_sp_name") or str(pid)),
                "team": g.get(side) or "",
                "opp": g.get(opp_side) or "",
                "gamePk": g.get("gamePk"),
                "last10": last10,
                "vs_top_offense": spl["hi"],
                "vs_bottom_offense": spl["lo"],
                **d,
            })
    # batters: recent regulars for every team on the slate
    team_ids = set()
    for g in slate:
        for k in ("away_id", "home_id"):
            if g.get(k):
                team_ids.add(g[k])
    opp_of = {}
    ab_map = {}
    team_pk = {}
    for g in slate:
        if g.get("away_id") and g.get("home_id"):
            opp_of[g["away_id"]] = g.get("home") or ""
            opp_of[g["home_id"]] = g.get("away") or ""
            ab_map[g["away_id"]] = g.get("away") or ""
            ab_map[g["home_id"]] = g.get("home") or ""
            team_pk[g["away_id"]] = g.get("gamePk")
            team_pk[g["home_id"]] = g.get("gamePk")
    batters = []
    if team_ids:
        bg = pd.read_sql(
            "SELECT batter_id, date, team_id, name, ab, h, hr, opp_id "
            "FROM batter_games "
            "WHERE ab IS NOT NULL AND ab > 0 AND date < ? ORDER BY date",
            con, params=(date,))
        lg_hit = (float(bg.h.sum()) / float(bg.ab.sum())
                  if len(bg) else LG_HIT_RATE_FALLBACK)
        lg_hr = (float(bg.hr.sum()) / float(bg.ab.sum())
                 if len(bg) else LG_HR_RATE_FALLBACK)
        hist2 = defaultdict(list)
        latest_team = {}
        latest_name = {}
        recent_ct = defaultdict(int)
        cutoff = (pd.Timestamp(date) - pd.Timedelta(days=14)).date().isoformat()
        bfull = defaultdict(list)
        for r in bg.itertuples():
            hist2[r.batter_id].append((r.h, r.hr, r.ab))
            bfull[r.batter_id].append(
                (r.date, int(r.ab), int(r.h), int(r.hr),
                 int(r.opp_id) if pd.notna(r.opp_id) else None))
            latest_team[r.batter_id] = r.team_id
            latest_name[r.batter_id] = r.name
            if r.date >= cutoff:
                recent_ct[r.batter_id] += 1
        per_team = defaultdict(list)
        for bid, tid in latest_team.items():
            if tid in team_ids and recent_ct.get(bid, 0) >= 5:
                per_team[tid].append(bid)
        for tid, bids in per_team.items():
            bids.sort(key=lambda b: -recent_ct[b])
            for bid in bids[:10]:
                pr = hist2[bid]
                hd = hit_distribution([(h, ab) for h, _, ab in pr], lg_hit)
                rd = hr_distribution([(hr, ab) for _, hr, ab in pr], lg_hr)
                if hd and rd:
                    fullb = bfull.get(bid, [])
                    last10 = [{"date": dt, "ab": ab_, "h": h_, "hr": hr_}
                              for dt, ab_, h_, hr_, _ in fullb[-10:]][::-1]
                    hs = _split([(h_, ab_, oid)
                                 for _, ab_, h_, hr_, oid in fullb],
                                pitch_q, pitch_top)
                    batters.append({
                        "batter_id": bid,
                        "name": latest_name.get(bid) or str(bid),
                        "team": ab_map.get(tid, ""),
                        "opp": opp_of.get(tid, ""),
                        "gamePk": team_pk.get(tid),
                        "last10": last10,
                        "vs_top_pitching": hs["hi"],
                        "vs_bottom_pitching": hs["lo"],
                        "hits": {"mean": hd["mean"], "over": hd["over"]},
                        "hr": {"mean": rd["mean"], "over": rd["over"]},
                        "n_prior": hd["n_prior"]})
    return {"date": date, "market": "strikeouts", "model": "props-k-v1",
            "pitchers": out, "batters": batters}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="slate.db")
    p.add_argument("--backtest", type=int, default=None,
                   help="season to walk point-in-time (strikeouts)")
    p.add_argument("--backtest-batting", type=int, default=None,
                   help="season to walk point-in-time (hits + HR)")
    p.add_argument("--out", default=None,
                   help="write JSON here (backtest or projections)")
    p.add_argument("--project", default=None,
                   help="date YYYY-MM-DD: project today's probable starters")
    args = p.parse_args()
    con = sqlite3.connect(args.db)
    if args.backtest_batting:
        res = backtest_batting(con, args.backtest_batting)
        print_batting(res)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(res, f, indent=1)
            print(f"written -> {args.out}")
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
