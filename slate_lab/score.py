"""Nightly job (step 3): refresh data, retrain, score today's slate.

Emits predictions.json shaped for the PWA:

{
  "generated": "...", "model": "logistic",
  "games": [{"gamePk": ..., "away": "NYY", "home": "CWS", "pAway": 0.548}]
}

The PWA fetches this from wherever the workflow publishes it (GitHub Pages
by default) and shows the ML number alongside the built-in model.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json

import pandas as pd

from .features import build_features, feature_columns
from .ingest import API, _get, connect, ingest_season
from .sports import get_sport
from .models import CLIP, make_gbdt, make_logistic, metrics


def todays_slate(date: str) -> list[dict]:
    j = _get(f"{API}/schedule?sportId=1&date={date}"
             f"&hydrate=probablePitcher,team")
    out = []
    for d in j.get("dates", []):
        for g in d.get("games", []):
            out.append({
                "gamePk": g["gamePk"],
                "away": g["teams"]["away"]["team"].get("abbreviation"),
                "home": g["teams"]["home"]["team"].get("abbreviation"),
                "away_id": g["teams"]["away"]["team"]["id"],
                "home_id": g["teams"]["home"]["team"]["id"],
                "away_sp": (g["teams"]["away"].get("probablePitcher") or {}).get("id"),
                "home_sp": (g["teams"]["home"].get("probablePitcher") or {}).get("id"),
            })
    return out


def slate_features(con, slate: list[dict], date: str) -> pd.DataFrame:
    """Reuse the historical builder by appending today's games as unplayed
    placeholders — build_features scores each game before folding its result,
    so a fake result on today's row never contaminates today's features."""
    season = int(date[:4])
    rows = [(10_000_000 + i, date, season, s["away_id"], s["home_id"],
             0, 1, s["away_sp"], s["home_sp"])  # placeholder result, see note
            for i, s in enumerate(slate)]
    con.executemany("INSERT OR REPLACE INTO games VALUES (?,?,?,?,?,?,?,?,?)",
                    rows)
    try:
        df = build_features(con)
        return df[df.game_pk >= 10_000_000].copy()
    finally:
        con.execute("DELETE FROM games WHERE game_pk >= 10000000")
        con.commit()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sport", default="mlb")
    p.add_argument("--db", default=None)
    p.add_argument("--date", default=dt.date.today().isoformat())
    p.add_argument("--update-season", action="store_true",
                   help="re-ingest the current season before scoring")
    p.add_argument("--out", default="predictions.json")
    args = p.parse_args()

    sport = get_sport(args.sport)
    db = args.db or (sport.key + ".db" if sport.key != "mlb" else "slate.db")
    con = connect(db)
    season = int(args.date[:4])
    if args.update_season:
        print(f"[{sport.name}] Refreshing {season}…")
        sport.ingest(con, season)

    hist = sport.build_features(con)
    train = hist[hist.date < args.date]
    cols = [c for g in sport.GROUPS for c in sport.GROUPS[g]]
    X, y = train[cols].to_numpy(), train["away_won"].to_numpy()

    lr = make_logistic().fit(X, y)
    gb = make_gbdt().fit(X, y)
    # pick by in-sample recency split: last 15% of training rows as a check
    cut = int(len(train) * 0.85)
    pick = min(
        [("logistic", lr), ("gbdt", gb)],
        key=lambda t: metrics(y[cut:], t[1].predict_proba(X[cut:])[:, 1])["logloss"],
    )
    name, model = pick

    slate = sport.slate(args.date)
    if not slate:
        json.dump({"generated": dt.datetime.utcnow().isoformat() + "Z",
                   "model": name, "games": []}, open(args.out, "w"))
        print("No games today.")
        return

    sf = sport.slate_features(con, slate, args.date)
    preds = model.predict_proba(sf[cols].to_numpy())[:, 1].clip(*CLIP)
    by_pk = dict(zip(sf.game_pk, preds))

    games = []
    for i, s in enumerate(slate):
        pk = 10_000_000 + i
        if pk in by_pk:
            games.append({"gamePk": s["gamePk"], "away": s["away"],
                          "home": s["home"],
                          "pAway": round(float(by_pk[pk]), 4)})

    json.dump({"generated": dt.datetime.utcnow().isoformat() + "Z",
               "sport": sport.key,
               "model": name, "trainRows": int(len(train)),
               "games": games}, open(args.out, "w"), indent=2)
    print(f"{len(games)} games scored with {name} → {args.out}")


if __name__ == "__main__":
    main()
