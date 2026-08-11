"""NFL yardage props: distribution sanity + point-in-time backtest on a
synthetic league with known per-player yardage processes."""
import random
import sqlite3
import sys

import numpy as np

sys.path.insert(0, ".")
from slate_lab.nfl_props import SCHEMA, yards_distribution, backtest


def main():
    rng = np.random.default_rng(4)
    pos_pool = rng.gamma(2.2, 22, size=2000)     # realistic right-skewed yards
    d = yards_distribution(
        [50, 60, 45, 70, 30, 55, 62, 48, 90, 20], pos_pool)
    assert d is not None
    q = d["q"]
    assert q["10"] <= q["25"] <= q["50"] <= q["75"] <= q["90"]
    assert d["over"]["25"] >= d["over"]["50"] >= d["over"]["75"] >= d["over"]["100"]
    assert yards_distribution([50] * 5, pos_pool) is None, "thin refused"
    print(f"  distribution: median {q['50']}, P(over 50)={d['over']['50']}")

    # synthetic league: gamma yards per player, walk one holdout season
    con = sqlite3.connect(":memory:")
    con.executescript(SCHEMA)
    pyr = random.Random(9)
    rows = []
    for pid in range(60):
        shape = max(1.2, pyr.gauss(2.2, 0.4))
        scale = max(8, pyr.gauss(24, 6))
        for season in (2023, 2024, 2025):
            for wk in range(1, 18):
                y = float(np.random.default_rng(pid * 1000 + season * 50 + wk)
                          .gamma(shape, scale))
                rows.append((f"P{pid}", season, wk, f"Player {pid}", "WR",
                             6, 4, y, 0, 0))
    con.executemany(
        "INSERT OR REPLACE INTO nfl_player_weeks VALUES (?,?,?,?,?,?,?,?,?,?)",
        rows)
    con.commit()
    import contextlib, io as _io
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        backtest(con, 2025)
    out = buf.getvalue()
    print("  " + out.strip().splitlines()[0])
    import re
    covs = {m[0]: float(m[1]) for m in
            re.findall(r"q(\d+)=([\d.]+)%", out)}
    for target in (10, 25, 50, 75, 90):
        assert abs(covs[str(target)] - target) < 8, (target, covs)
    print("  synthetic coverage within 8pts of nominal at all quantiles")
    print("\nNFL PROPS TESTS PASSED")


if __name__ == "__main__":
    main()
