"""Strikeout distribution tests: shape, point-in-time, calibration on
a synthetic league where true K rates are known."""
import random
import sqlite3
import sys

import numpy as np

sys.path.insert(0, ".")
from slate_lab.props import k_distribution, backtest, MAX_K


def synth(seed=5):
    con = sqlite3.connect(":memory:")
    con.executescript("""
    CREATE TABLE pitcher_starts (
      pitcher_id INTEGER, date TEXT, season INTEGER,
      er INTEGER, outs INTEGER, so INTEGER, bb INTEGER, bf INTEGER,
      PRIMARY KEY (pitcher_id, date));""")
    rng = random.Random(seed)
    rows = []
    for pid in range(60):
        true_rate = min(0.38, max(0.10, rng.gauss(0.22, 0.05)))
        for season in (2023, 2024, 2025):
            for i, day in enumerate(range(1, 28)):
                bf = max(12, round(rng.gauss(23, 3)))
                so = sum(1 for _ in range(bf) if rng.random() < true_rate)
                rows.append((pid, f"{season}-{4 + i % 6:02d}-{day:02d}",
                             season, 2, 17, so, 2, bf))
    con.executemany("INSERT OR REPLACE INTO pitcher_starts VALUES "
                    "(?,?,?,?,?,?,?,?)", rows)
    con.commit()
    return con


def main():
    # distribution basics
    prior = [(6, 24), (7, 22), (5, 25), (8, 23)]
    d = k_distribution(prior, 0.22)
    assert d is not None
    pmf = np.array(d["pmf"])
    assert abs(pmf.sum() - 1.0) < 1e-3, "pmf must sum to 1 (5dp storage)"
    assert 0 < d["mean"] < MAX_K
    assert d["over"]["5.5"] > d["over"]["7.5"], "over probs must decrease"
    print(f"  distribution: mean {d['mean']}, P(over 5.5)={d['over']['5.5']}")

    assert k_distribution([(6, 24)], 0.22) is None, "thin history -> None"
    print("  thin history refused")

    con = synth()
    res = backtest(con, 2025)
    assert res["n"] > 1000
    assert res["logscore"] < res["logscore_naive"], (
        f"distribution {res['logscore']} should beat naive "
        f"{res['logscore_naive']}")
    # calibration: within 10 points on synthetic data. The synthetic world
    # has STATIC true rates, so the league-regression prior (correct for
    # real pitchers, who genuinely regress season to season) reads as a
    # mild centering bias here. 10pts allows that known artifact; the
    # real-data backtest is the calibration that actually matters.
    for line, buckets in res["calibration"].items():
        for b in buckets:
            assert abs(b["pred"] - b["hit"]) < 0.10, (line, b)
    print(f"  backtest: n={res['n']}, log {res['logscore']} vs naive "
          f"{res['logscore_naive']}, calibration within 10pts everywhere")
    print("\nPROPS TESTS PASSED")


if __name__ == "__main__":
    main()
