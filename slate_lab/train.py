"""Walk-forward training and the ablation loop.

Validation is BY SEASON: train on all seasons before the test season, never
after it. Random K-fold on time series data smuggles the future into the
past and inflates every number; season splits are the honest version.

The ablation is the "feedback loop" done properly: each feature group is
added cumulatively and must improve held-out log loss to earn its keep.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .features import GROUPS, build_features, feature_columns
from .ingest import connect
from .models import (baseline_predict, calibration_table, make_gbdt,
                     make_logistic, metrics)


def walk_forward(df: pd.DataFrame, test_season: int, groups: list[str]):
    cols = feature_columns(groups)
    train = df[df.season < test_season]
    test = df[df.season == test_season]
    if len(train) < 300 or len(test) < 100:
        raise SystemExit(
            f"Not enough data: train={len(train)} test={len(test)}. "
            "Ingest more seasons first.")
    Xtr, ytr = train[cols].to_numpy(), train["away_won"].to_numpy()
    Xte, yte = test[cols].to_numpy(), test["away_won"].to_numpy()

    out = {"test_season": test_season, "groups": groups,
           "n_train": len(train), "n_test": len(test)}

    out["baseline"] = metrics(yte, baseline_predict(test))

    lr = make_logistic().fit(Xtr, ytr)
    p_lr = lr.predict_proba(Xte)[:, 1]
    out["logistic"] = metrics(yte, p_lr)

    gb = make_gbdt().fit(Xtr, ytr)
    p_gb = gb.predict_proba(Xte)[:, 1]
    out["gbdt"] = metrics(yte, p_gb)

    best = min(("logistic", p_lr), ("gbdt", p_gb),
               key=lambda t: metrics(yte, t[1])["logloss"])
    out["best_model"] = best[0]
    out["calibration"] = calibration_table(yte, best[1])
    return out


def correlations(df: pd.DataFrame) -> dict:
    """Feature-vs-outcome and feature-vs-feature correlations.

    The second matters as much as the first: a feature can correlate with
    winning and still add nothing, if it correlates just as hard with a
    feature already in the model."""
    cols = feature_columns(list(GROUPS))
    num = [c for c in cols if df[c].nunique() > 2]
    return {
        "with_outcome": df[num].corrwith(df["away_won"]).round(4).to_dict(),
        "with_pyth_diff": df[num].corrwith(df["pyth_diff"]).round(4).to_dict(),
    }


def ablation(df: pd.DataFrame, test_season: int):
    """Add feature groups one at a time; report held-out log loss at each step."""
    steps, active = [], []
    for grp in GROUPS:
        active.append(grp)
        r = walk_forward(df, test_season, list(active))
        steps.append({
            "added": grp, "groups": list(active),
            "logistic_logloss": r["logistic"]["logloss"],
            "gbdt_logloss": r["gbdt"]["logloss"],
        })
    return steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="slate.db")
    p.add_argument("--test-season", type=int, required=True)
    p.add_argument("--ablation", action="store_true")
    p.add_argument("--out", default="results.json")
    args = p.parse_args()

    con = connect(args.db)
    print("Building point-in-time features…")
    df = build_features(con)
    print(f"  {len(df)} scoreable games "
          f"across seasons {sorted(df.season.unique().tolist())}")

    result = walk_forward(df, args.test_season, list(GROUPS))
    result["correlations"] = correlations(df)
    if args.ablation:
        result["ablation"] = ablation(df, args.test_season)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== Test season {args.test_season} "
          f"(n={result['n_test']}) ===")
    for name in ("baseline", "logistic", "gbdt"):
        m = result[name]
        print(f"  {name:9s} logloss={m['logloss']:.4f} "
              f"brier={m['brier']:.4f} skill={m['skill']:+.1%}")
    c = result["correlations"]
    print("\n  Correlation with outcome / with pyth_diff:")
    for k in c["with_outcome"]:
        print(f"    {k:12s} {c['with_outcome'][k]:+.4f}  "
              f"{c['with_pyth_diff'][k]:+.4f}")
    if "ablation" in result:
        print("\n  Ablation (held-out logloss as groups are added):")
        for s in result["ablation"]:
            print(f"    +{s['added']:8s} lr={s['logistic_logloss']:.4f} "
                  f"gbdt={s['gbdt_logloss']:.4f}")
    print(f"\nFull results → {args.out}")


if __name__ == "__main__":
    main()
