"""Pipeline tests on a synthetic league with KNOWN ground truth.

The synthetic league gives us something real data never can: we know the
true win probabilities, so we know what a correct model SHOULD score.

The tampering test is the one that matters. If any feature leaks future
information, changing a late-season result would change early-season
features — so we flip one and demand that nothing before it moves.
"""
import math
import random
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from slate_lab.features import GROUPS, build_features, feature_columns
from slate_lab.ingest import SCHEMA
from slate_lab.models import baseline_predict, make_gbdt, make_logistic, metrics
from slate_lab.train import walk_forward


def synth_league(seed=11, seasons=(2023, 2024, 2025), n_teams=30,
                 games_per_team=140):
    rng = random.Random(seed)
    con = sqlite3.connect(":memory:")
    con.executescript(SCHEMA)
    pk = 1
    for season in seasons:
        strength = {t: rng.gauss(0, 0.35) for t in range(1, n_teams + 1)}
        # each team gets a stable of 5 starters with their own quality
        arms = {t: [(t * 100 + k, rng.gauss(strength[t] * 0.3, 0.25))
                    for k in range(5)] for t in strength}
        rot = {t: 0 for t in strength}
        date0 = pd.Timestamp(f"{season}-04-01")
        n_games = n_teams * games_per_team // 2
        for i in range(n_games):
            date = (date0 + pd.Timedelta(days=i // 15)).strftime("%Y-%m-%d")
            a, h = rng.sample(list(strength), 2)
            asp_id, asp_q = arms[a][rot[a] % 5]
            hsp_id, hsp_q = arms[h][rot[h] % 5]
            rot[a] += 1
            rot[h] += 1
            z = (strength[a] - strength[h]) + 0.7 * (asp_q - hsp_q) - 0.17
            p_away = 1 / (1 + math.exp(-z))
            away_won = rng.random() < p_away
            asc, hsc = (5, 3) if away_won else (3, 5)
            asc += max(0, round(rng.gauss(strength[a] * 2, 1)))
            hsc += max(0, round(rng.gauss(strength[h] * 2, 1)))
            if asc == hsc:
                (asc, hsc) = (asc + 1, hsc) if away_won else (asc, hsc + 1)
            con.execute("INSERT INTO games VALUES (?,?,?,?,?,?,?,?,?)",
                        (pk, date, season, a, h, asc, hsc, asp_id, hsp_id))
            # starter line correlated with quality
            er = max(0, round(rng.gauss(3.5 - asp_q * 2.5, 1.4)))
            con.execute("INSERT OR REPLACE INTO pitcher_starts VALUES (?,?,?,?,?)",
                        (asp_id, date, season, er, 17))
            er = max(0, round(rng.gauss(3.5 - hsp_q * 2.5, 1.4)))
            con.execute("INSERT OR REPLACE INTO pitcher_starts VALUES (?,?,?,?,?)",
                        (hsp_id, date, season, er, 17))
            pk += 1
    con.commit()
    return con


def test_features_build():
    con = synth_league()
    df = build_features(con)
    assert len(df) > 3000, f"too few scoreable rows: {len(df)}"
    assert df["away_won"].mean() > 0.35 and df["away_won"].mean() < 0.55
    assert not df[feature_columns()].isna().any().any(), "NaNs in features"
    print(f"  features: {len(df)} rows, "
          f"away win rate {df['away_won'].mean():.3f}")


def test_no_leakage_tripwire():
    """Flip one late-season result; every feature row before that date
    must be byte-identical. If this ever fails, stop everything."""
    con = synth_league()
    before = build_features(con)

    last = con.execute(
        "SELECT game_pk, date, away_score, home_score FROM games "
        "WHERE season=2025 ORDER BY date DESC LIMIT 1").fetchone()
    pk, tamper_date, a, h = last
    con.execute("UPDATE games SET away_score=?, home_score=? WHERE game_pk=?",
                (h, a, pk))  # flip the result
    after = build_features(con)

    cols = feature_columns() + ["away_won"]
    b = before[before.date < tamper_date].reset_index(drop=True)
    f = after[after.date < tamper_date].reset_index(drop=True)
    pd.testing.assert_frame_equal(b[cols], f[cols])
    print(f"  tripwire: flipped game {pk} on {tamper_date}; "
          f"{len(b)} earlier rows unchanged ✓")


def test_models_beat_chance_and_recover_signal():
    con = synth_league()
    df = build_features(con)
    r = walk_forward(df, test_season=2025, groups=list(GROUPS))
    print(f"  baseline logloss {r['baseline']['logloss']:.4f} | "
          f"logistic {r['logistic']['logloss']:.4f} | "
          f"gbdt {r['gbdt']['logloss']:.4f}")
    assert r["logistic"]["logloss"] < 0.693, "worse than a coin flip"
    assert r["logistic"]["skill"] > 0.02, "no skill recovered from known signal"
    # trained models should beat the fixed-weight baseline on synthetic data
    assert r["logistic"]["logloss"] <= r["baseline"]["logloss"] + 0.01


def test_calibration_sane():
    con = synth_league()
    df = build_features(con)
    r = walk_forward(df, 2025, list(GROUPS))
    for b in r["calibration"]:
        gap = abs(b["predicted"] - b["actual"])
        se = (b["actual"] * (1 - b["actual"]) / b["n"]) ** 0.5 if b["n"] else 1
        assert gap < max(0.12, 3.5 * se), f"badly miscalibrated bucket: {b}"
    print(f"  calibration: {len(r['calibration'])} buckets within tolerance")


if __name__ == "__main__":
    for fn in (test_features_build, test_no_leakage_tripwire,
               test_models_beat_chance_and_recover_signal,
               test_calibration_sane):
        print(fn.__name__)
        fn()
    print("\nALL TESTS PASSED")
