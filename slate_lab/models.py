"""Three tiers of model, so every claim of improvement has a floor under it.

  baseline  — the closed-form model the PWA already runs (log5 + HFA + starter)
  logistic  — linear model over the feature table
  gbdt      — HistGradientBoosting, small and regularized on purpose

If gbdt cannot beat logistic out of sample, the extra machinery is
memorizing noise and the honest answer is to ship the simpler model.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HFA_LOGIT = 0.155
TEAM_REGRESS = 0.85
RUNS_TO_LOGIT = 0.34
CLIP = (0.15, 0.85)


def baseline_predict(df: pd.DataFrame) -> np.ndarray:
    """The PWA model, reproduced exactly, from the feature table."""
    pyth_a = 0.5 + (df["pyth_diff"] / 2) * TEAM_REGRESS + 0.0
    # pyth_diff = pa - ph; reconstruct symmetric pair around .500
    pa = 0.5 + df["pyth_diff"] * TEAM_REGRESS / 2
    ph = 0.5 - df["pyth_diff"] * TEAM_REGRESS / 2
    den = pa + ph - 2 * pa * ph
    base = np.where(den == 0, 0.5, (pa - pa * ph) / den)
    z = np.log(np.clip(base, 0.05, 0.95) / (1 - np.clip(base, 0.05, 0.95)))
    z = z - HFA_LOGIT + df["sp_edge"].to_numpy() * RUNS_TO_LOGIT
    p = 1 / (1 + np.exp(-z))
    return np.clip(p, *CLIP)


def make_logistic() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.5, max_iter=2000),
    )


def make_gbdt() -> object:
    # deliberately small: shallow trees, strong regularization, few iterations —
    # a few hundred games per test season cannot support anything bigger
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=150,
        l2_regularization=1.0, min_samples_leaf=40,
        validation_fraction=0.15, early_stopping=True, random_state=7,
    )


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return {
        "n": int(len(y)),
        "brier": float(np.mean((p - y) ** 2)),
        "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "acc": float(np.mean((p >= 0.5) == (y == 1))),
        "skill": float(1 - np.mean((p - y) ** 2) / 0.25),
    }


def calibration_table(y: np.ndarray, p: np.ndarray, lo=0.20, hi=0.80, k=6):
    edges = np.linspace(lo, hi, k + 1)
    out = []
    for i in range(k):
        m = (p >= edges[i]) & (p < edges[i + 1])
        if m.sum() >= 15:
            out.append({
                "bucket": f"{edges[i]:.0%}–{edges[i+1]:.0%}",
                "n": int(m.sum()),
                "predicted": float(p[m].mean()),
                "actual": float(y[m].mean()),
            })
    return out
