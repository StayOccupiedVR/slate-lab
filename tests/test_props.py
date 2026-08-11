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
      opp_id INTEGER, PRIMARY KEY (pitcher_id, date));""")
    rng = random.Random(seed)
    rows = []
    for pid in range(60):
        true_rate = min(0.38, max(0.10, rng.gauss(0.22, 0.05)))
        for season in (2023, 2024, 2025):
            for i, day in enumerate(range(1, 28)):
                bf = max(12, round(rng.gauss(23, 3)))
                so = sum(1 for _ in range(bf) if rng.random() < true_rate)
                rows.append((pid, f"{season}-{4 + i % 6:02d}-{day:02d}",
                             season, 2, 17, so, 2, bf, 100 + (pid + i) % 4))
    con.executemany("INSERT OR REPLACE INTO pitcher_starts VALUES "
                    "(?,?,?,?,?,?,?,?,?)", rows)
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





def test_batting():
    """Hits + HR distributions on a synthetic league with known rates."""
    import sqlite3
    import random
    import numpy as np
    from slate_lab.props import (hit_distribution, hr_distribution,
                                 backtest_batting, project_slate)

    con = sqlite3.connect(":memory:")
    con.executescript("""
    CREATE TABLE batter_games (
      batter_id INTEGER, date TEXT, season INTEGER, team_id INTEGER,
      name TEXT, pa INTEGER, ab INTEGER, h INTEGER, hr INTEGER,
      opp_id INTEGER, PRIMARY KEY (batter_id, date));
    CREATE TABLE pitcher_starts (
      pitcher_id INTEGER, date TEXT, season INTEGER, er INTEGER,
      outs INTEGER, so INTEGER, bb INTEGER, bf INTEGER, opp_id INTEGER,
      PRIMARY KEY (pitcher_id, date));
    CREATE TABLE games (gamePk INTEGER PRIMARY KEY, date TEXT,
      season INTEGER, away_id INTEGER, home_id INTEGER,
      away_score INTEGER, home_score INTEGER,
      away_sp INTEGER, home_sp INTEGER);""")
    rng = random.Random(11)
    rows = []
    for bid in range(80):
        hit_rate = min(0.34, max(0.18, rng.gauss(0.25, 0.03)))
        hr_rate = min(0.09, max(0.005, rng.gauss(0.032, 0.015)))
        team = 100 + bid % 4
        for season in (2024, 2025):
            for i in range(120):
                ab = rng.choice([3, 3, 4, 4, 4, 5])
                h = sum(1 for _ in range(ab) if rng.random() < hit_rate)
                hr = sum(1 for _ in range(ab) if rng.random() < hr_rate)
                rows.append((bid, f"{season}-{4 + i // 26:02d}-{1 + i % 26:02d}",
                             season, team, f"Batter {bid}", ab + 1, ab, h,
                             min(hr, h), 100 + (bid + i) % 4))
    con.executemany(
        "INSERT OR REPLACE INTO batter_games VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    for gpk in range(1, 40):
        a, h = 100 + gpk % 4, 100 + (gpk + 1) % 4
        con.execute("INSERT INTO games VALUES (?,?,?,?,?,?,?,?,?)",
                    (gpk, f"2025-{4 + gpk // 26:02d}-{1 + gpk % 26:02d}", 2025,
                     a, h, 4, 3, None, None))
    con.commit()

    prior = [(1, 4)] * 40
    hd = hit_distribution(prior, 0.245)
    assert hd and hd["over"]["0.5"] > hd["over"]["1.5"] > hd["over"]["2.5"]
    assert hit_distribution([(1, 4)] * 10, 0.245) is None, "thin refused"
    rd = hr_distribution([(0, 4)] * 35 + [(1, 4)] * 5, 0.032)
    assert rd and rd["over"]["0.5"] < 0.5
    print(f"  distributions: P(1+ hit)={hd['over']['0.5']}, "
          f"P(HR)={rd['over']['0.5']}")

    res = backtest_batting(con, 2025)
    assert res["n"] > 5000
    for mk, buckets in res["calibration"].items():
        for b in buckets:
            assert abs(b["pred"] - b["hit"]) < 0.10, (mk, b)
    print(f"  batting backtest: n={res['n']}, calibration within 10pts")

    slate = [{"gamePk": 1, "away": "AAA", "home": "BBB",
              "away_id": 100, "home_id": 101,
              "away_sp": None, "home_sp": None}]
    doc = project_slate(con, slate, "2025-08-20")
    assert len(doc["batters"]) > 0
    b0 = doc["batters"][0]
    assert b0["team"] in ("AAA", "BBB") and "0.5" in b0["hits"]["over"]
    assert b0["opp"] in ("AAA", "BBB") and b0["opp"] != b0["team"]
    assert len(b0["last10"]) == 10 and {"date", "ab", "h", "hr"} <= set(b0["last10"][0])
    assert b0["last10"][0]["date"] > b0["last10"][-1]["date"], "newest first"
    assert "vs_top_pitching" in b0 and "vs_bottom_pitching" in b0
    assert (b0["vs_top_pitching"]["n"] + b0["vs_bottom_pitching"]["n"]) > 0
    print(f"  feed: {len(doc['batters'])} batters, last10 + splits enriched")
    print("\nBATTING TESTS PASSED")


if __name__ == "__main__":
    main()
    test_batting()
