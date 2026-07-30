"""Statcast team-quality features (step 2) — optional, additive, self-proving.

Pulls pitch-level Statcast data through pybaseball, aggregates to a
team-day table of rolling expected wOBA (xwOBA), and merges a
`xwoba_diff` feature into the frame with the same as-of discipline as
everything else: a game on date D sees only Statcast rows before D.

Why xwOBA: it strips defense and luck from batted-ball outcomes the way
FIP does for pitchers, and it stabilizes faster than actual results.

This module registers itself into features.GROUPS on import, so the
ablation in train.py will automatically tell you whether Statcast earned
its place. Run:

    python -m slate_lab.statcast --db slate.db --seasons 2024 2025
    python -m slate_lab.train --db slate.db --test-season 2025 --ablation

Heads-up: the Statcast download is large (a season is ~700k pitches) and
Baseball Savant rate-limits aggressively. pybaseball caches locally;
expect the first pull per season to take a while.
"""
from __future__ import annotations

import sqlite3

import pandas as pd

from . import features as F

SCHEMA = """
CREATE TABLE IF NOT EXISTS team_statcast_daily (
  team_id INTEGER NOT NULL,
  date    TEXT NOT NULL,
  xwoba_30d REAL,
  PRIMARY KEY (team_id, date)
);
"""

# Statcast names teams by abbreviation; the games table uses MLBAM ids.
TEAM_IDS = {
    "AZ": 109, "ARI": 109, "ATL": 144, "BAL": 110, "BOS": 111, "CHC": 112,
    "CWS": 145, "CHW": 145, "CIN": 113, "CLE": 114, "COL": 115, "DET": 116,
    "HOU": 117, "KC": 118, "KCR": 118, "LAA": 108, "LAD": 119, "MIA": 146,
    "MIL": 158, "MIN": 142, "NYM": 121, "NYY": 147, "OAK": 133, "ATH": 133,
    "PHI": 143, "PIT": 134, "SD": 135, "SDP": 135, "SEA": 136, "SF": 137,
    "SFG": 137, "STL": 138, "TB": 139, "TBR": 139, "TEX": 140, "TOR": 141,
    "WSH": 120, "WSN": 120,
}


def ingest_statcast(con: sqlite3.Connection, season: int) -> None:
    from pybaseball import statcast  # heavy import kept local

    con.executescript(SCHEMA)
    raw = statcast(start_dt=f"{season}-03-15", end_dt=f"{season}-10-05")
    bat = raw.dropna(subset=["estimated_woba_using_speedangle"])[
        ["game_date", "bat_team" if "bat_team" in raw.columns else "home_team",
         "estimated_woba_using_speedangle"]
    ].copy()
    bat.columns = ["date", "team", "xwoba"]
    bat["date"] = pd.to_datetime(bat["date"]).dt.strftime("%Y-%m-%d")
    bat["team_id"] = bat["team"].map(TEAM_IDS)
    bat = bat.dropna(subset=["team_id"])

    daily = (bat.groupby(["team_id", "date"])["xwoba"]
                .agg(["sum", "count"]).reset_index())
    rows = []
    for tid, g in daily.groupby("team_id"):
        g = g.sort_values("date")
        # 30-day trailing mean, shifted one day so date D excludes D itself
        s = g.set_index(pd.to_datetime(g["date"]))
        roll = (s["sum"].rolling("30D").sum().shift(1)
                / s["count"].rolling("30D").sum().shift(1))
        for d, v in zip(g["date"], roll):
            if pd.notna(v):
                rows.append((int(tid), d, float(v)))
    con.executemany(
        "INSERT OR REPLACE INTO team_statcast_daily VALUES (?,?,?)", rows)
    con.commit()
    print(f"  statcast {season}: {len(rows)} team-day rows")


def merge_xwoba(con: sqlite3.Connection, df: pd.DataFrame) -> pd.DataFrame:
    """Attach xwoba_diff to the feature frame. As-of by construction:
    the daily table's rolling window is already shifted one day."""
    try:
        sc = pd.read_sql("SELECT * FROM team_statcast_daily", con)
    except Exception:
        df["xwoba_diff"] = 0.0
        return df
    if sc.empty:
        df["xwoba_diff"] = 0.0
        return df

    games = pd.read_sql("SELECT game_pk, away_id, home_id FROM games", con)
    df = df.merge(games, on="game_pk", how="left")
    for side in ("away", "home"):
        df = df.merge(
            sc.rename(columns={"team_id": f"{side}_id",
                               "xwoba_30d": f"xwoba_{side}"}),
            left_on=[f"{side}_id", "date"],
            right_on=[f"{side}_id", "date"], how="left")
    df["xwoba_diff"] = (df["xwoba_away"].fillna(0)
                        - df["xwoba_home"].fillna(0))
    return df.drop(columns=["away_id", "home_id",
                            "xwoba_away", "xwoba_home"])


# register the group so the ablation harness sees it
F.GROUPS["statcast"] = ["xwoba_diff"]

_orig_build = F.build_features


def _build_with_statcast(con, **kw):
    return merge_xwoba(con, _orig_build(con, **kw))


F.build_features = _build_with_statcast


if __name__ == "__main__":
    import argparse
    from .ingest import connect
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="slate.db")
    p.add_argument("--seasons", type=int, nargs="+", required=True)
    a = p.parse_args()
    con = connect(a.db)
    for s in a.seasons:
        print(f"Statcast {s}…")
        ingest_statcast(con, s)
