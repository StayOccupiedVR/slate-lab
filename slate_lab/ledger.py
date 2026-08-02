"""Prediction ledger + sportsbook odds capture (The Odds API).

File-based and append-only on purpose: every snapshot and prediction is
committed to git by the workflow, so commit timestamps prove predictions
existed before first pitch. A database row can be rewritten quietly; a
git history can't.

Books: DraftKings preferred, FanDuel then Hard Rock Bet as per-game
fallbacks — all three captured in every snapshot for price comparison. DraftKings is licensed in more states and carries
deeper liquidity, which makes its closing line the better benchmark. Both
live in The Odds API's US regions.
Free tier = 500 credits/month; each capture costs regions x markets
(2 x 1 = 2), so three captures a day ≈ 180/month. Comfortable.

Commands:
  python -m slate_lab.ledger capture-odds   # snapshot current moneylines
  python -m slate_lab.ledger record         # file today's predictions.json
  python -m slate_lab.ledger grade          # grade finished dates, write report
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import math
import os
import urllib.request
from pathlib import Path

from .ingest import API as MLB_API
from .ingest import _get

ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports/{sport}/odds"
BOOKS = ["draftkings", "fanduel", "hardrockbet"]   # preference order everywhere
DATA = Path("data")

def _configure(sport_key: str):
    """Point module globals at a sport's odds feed, books, paths, and team map."""
    global ODDS_API, BOOKS, DATA, TEAM_NAME_TO_ID
    from .sports import get_sport
    sp = get_sport(sport_key)
    ODDS_API = ODDS_API_BASE.format(sport=sp.odds_sport)
    BOOKS = list(sp.books)
    DATA = Path("data") / sp.data_prefix if sp.data_prefix else Path("data")
    if sp.team_name_to_id:
        TEAM_NAME_TO_ID = sp.team_name_to_id
    return sp

ODDS_API = ODDS_API_BASE.format(sport="baseball_mlb")

TEAM_NAME_TO_ID = {
    "Arizona Diamondbacks": 109, "Atlanta Braves": 144, "Baltimore Orioles": 110,
    "Boston Red Sox": 111, "Chicago Cubs": 112, "Chicago White Sox": 145,
    "Cincinnati Reds": 113, "Cleveland Guardians": 114, "Colorado Rockies": 115,
    "Detroit Tigers": 116, "Houston Astros": 117, "Kansas City Royals": 118,
    "Los Angeles Angels": 108, "Los Angeles Dodgers": 119, "Miami Marlins": 146,
    "Milwaukee Brewers": 158, "Minnesota Twins": 142, "New York Mets": 121,
    "New York Yankees": 147, "Oakland Athletics": 133, "Athletics": 133,
    "Philadelphia Phillies": 143, "Pittsburgh Pirates": 134,
    "San Diego Padres": 135, "San Francisco Giants": 137, "Seattle Mariners": 136,
    "St. Louis Cardinals": 138, "St Louis Cardinals": 138, "Tampa Bay Rays": 139,
    "Texas Rangers": 140, "Toronto Blue Jays": 141, "Washington Nationals": 120,
}


def ml_to_prob(ml: float) -> float:
    return 100 / (ml + 100) if ml > 0 else abs(ml) / (abs(ml) + 100)


def devig(p_a: float, p_h: float) -> tuple[float, float]:
    s = p_a + p_h
    return (p_a / s, p_h / s) if s > 0 else (0.5, 0.5)


# ---------------------------------------------------------------- capture
def capture_odds(api_key: str) -> Path:
    url = (f"{ODDS_API}?apiKey={api_key}&regions=us,us2&markets=h2h"
           f"&oddsFormat=american")
    with urllib.request.urlopen(url, timeout=30) as r:
        events = json.loads(r.read().decode())

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rows = []
    for ev in events:
        away_id = TEAM_NAME_TO_ID.get(ev.get("away_team"))
        home_id = TEAM_NAME_TO_ID.get(ev.get("home_team"))
        if not away_id or not home_id:
            continue
        row = {"commence": ev["commence_time"], "away_id": away_id,
               "home_id": home_id, "away": ev["away_team"],
               "home": ev["home_team"], "books": {}}
        for bk in ev.get("bookmakers", []):
            if bk["key"] not in BOOKS:
                continue
            m = next((m for m in bk.get("markets", []) if m["key"] == "h2h"), None)
            if not m:
                continue
            prices = {o["name"]: o["price"] for o in m.get("outcomes", [])}
            if ev["away_team"] in prices and ev["home_team"] in prices:
                row["books"][bk["key"]] = {
                    "ml_away": prices[ev["away_team"]],
                    "ml_home": prices[ev["home_team"]],
                }
        if row["books"]:
            rows.append(row)

    out = DATA / "odds" / f"{ts}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"captured": ts, "events": rows}, indent=1))
    print(f"{len(rows)} games with odds -> {out}")
    return out


def export_odds(out_file: str = "odds.json") -> Path | None:
    """Flatten the latest snapshot into a small file the app can fetch.

    One row per upcoming game with the preferred book's current moneylines.
    Published to gh-pages next to predictions.json so game cards can show
    live DraftKings prices without any additional API spend."""
    snaps = sorted((DATA / "odds").glob("*.json")) if (DATA / "odds").exists() else []
    if not snaps:
        print("No snapshots to export.")
        return None
    snap = json.loads(snaps[-1].read_text())
    rows = []
    for ev in snap.get("events", []):
        book = next((b for b in BOOKS if b in ev.get("books", {})), None)
        if not book:
            continue
        rows.append({
            "away_id": ev["away_id"], "home_id": ev["home_id"],
            "commence": ev["commence"], "book": book,
            "ml_away": ev["books"][book]["ml_away"],
            "ml_home": ev["books"][book]["ml_home"],
            "books": ev["books"],
        })
    out = Path(out_file)
    out.write_text(json.dumps({
        "captured": snap.get("captured"), "games": rows}, indent=1))
    print(f"{len(rows)} games -> {out}")
    return out


# ---------------------------------------------------------------- record
def record_predictions(pred_file: str = "predictions.json") -> Path | None:
    p = json.loads(Path(pred_file).read_text())
    if not p.get("games"):
        print("No games in predictions.json; nothing to record.")
        return None
    date = dt.date.today().isoformat()
    out = DATA / "predictions" / f"{date}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"{out} already exists — refusing to overwrite a filed prediction.")
        return out
    p["recorded"] = dt.datetime.now(dt.timezone.utc).isoformat()
    out.write_text(json.dumps(p, indent=1))
    print(f"Filed {len(p['games'])} predictions -> {out}")
    return out


# ---------------------------------------------------------------- grade
def _closing_odds() -> dict:
    """(date, away_id, home_id) -> devigged away prob from the snapshot
    closest to (but before) first pitch, preferring DraftKings."""
    best: dict = {}
    for f in sorted(glob.glob(str(DATA / "odds" / "*.json"))):
        snap = json.loads(Path(f).read_text())
        cap = snap["captured"]
        for ev in snap["events"]:
            if cap > ev["commence"].replace("-", "").replace(":", ""):
                continue  # captured after first pitch; not a closing number
            key = (ev["commence"][:10], ev["away_id"], ev["home_id"])
            book = next((b for b in BOOKS if b in ev["books"]), None)
            if not book:
                continue
            prev = best.get(key)
            if prev is None or cap > prev["cap"]:
                o = ev["books"][book]
                pa, ph = devig(ml_to_prob(o["ml_away"]), ml_to_prob(o["ml_home"]))
                best[key] = {"cap": cap, "book": book, "p_away": pa,
                             "ml_away": o["ml_away"], "ml_home": o["ml_home"]}
    return best


def grade() -> None:
    closing = _closing_odds()
    graded_dir = DATA / "graded"
    graded_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for pf in sorted(glob.glob(str(DATA / "predictions" / "*.json"))):
        date = Path(pf).stem
        gf = graded_dir / f"{date}.json"
        if gf.exists():
            all_rows.extend(json.loads(gf.read_text())["games"])
            continue
        if date >= dt.date.today().isoformat():
            continue  # not finished yet

        preds = json.loads(Path(pf).read_text())
        sched = _get(f"{MLB_API}/schedule?sportId=1&date={date}&hydrate=team")
        finals = {}
        for d in sched.get("dates", []):
            for g in d.get("games", []):
                if g.get("status", {}).get("abstractGameState") != "Final":
                    continue
                a, h = g["teams"]["away"], g["teams"]["home"]
                if a.get("score") is None or a["score"] == h["score"]:
                    continue
                finals[g["gamePk"]] = {
                    "away_won": a["score"] > h["score"],
                    "away_id": a["team"]["id"], "home_id": h["team"]["id"],
                }

        rows = []
        for g in preds["games"]:
            f = finals.get(g["gamePk"])
            if not f:
                continue
            mkt = closing.get((date, f["away_id"], f["home_id"]))
            rows.append({
                "date": date, "gamePk": g["gamePk"],
                "away": g["away"], "home": g["home"],
                "p_model": g["pAway"], "away_won": f["away_won"],
                "p_market": mkt["p_away"] if mkt else None,
                "book": mkt["book"] if mkt else None,
            })
        if rows:
            gf.write_text(json.dumps(
                {"date": date, "model": preds.get("model"), "games": rows},
                indent=1))
            all_rows.extend(rows)
            print(f"graded {date}: {len(rows)} games")

    _report(all_rows)


def _ll(y: int, p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def _report(rows: list[dict]) -> None:
    if not rows:
        print("Nothing graded yet.")
        return
    n = len(rows)
    ll_model = sum(_ll(r["away_won"], r["p_model"]) for r in rows) / n
    acc = sum((r["p_model"] >= 0.5) == r["away_won"] for r in rows) / n
    rep = {"generated": dt.datetime.now(dt.timezone.utc).isoformat(),
           "n_games": n, "model_logloss": round(ll_model, 4),
           "model_accuracy": round(acc, 4)}

    mkt = [r for r in rows if r["p_market"] is not None]
    if mkt:
        m = len(mkt)
        rep["n_with_odds"] = m
        rep["market_logloss"] = round(
            sum(_ll(r["away_won"], r["p_market"]) for r in mkt) / m, 4)
        rep["model_logloss_on_same"] = round(
            sum(_ll(r["away_won"], r["p_model"]) for r in mkt) / m, 4)
        # CLV proxy: when model disagreed with market side, who was right?
        dis = [r for r in mkt if (r["p_model"] >= 0.5) != (r["p_market"] >= 0.5)]
        if dis:
            rep["disagreements"] = len(dis)
            rep["model_won_disagreements"] = round(
                sum((r["p_model"] >= 0.5) == r["away_won"] for r in dis)
                / len(dis), 4)

    out = DATA / "report.json"
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


# ---------------------------------------------------------------- cli
def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["capture-odds", "record", "grade", "export-odds"])
    p.add_argument("--sport", default="mlb")
    p.add_argument("--pred", default="predictions.json")
    args = p.parse_args()
    _configure(args.sport)
    if args.cmd == "export-odds":
        export_odds()
        return
    if args.cmd == "capture-odds":
        key = os.environ.get("ODDS_API_KEY")
        if not key:
            raise SystemExit(
                "Set ODDS_API_KEY (free key: the-odds-api.com).")
        capture_odds(key)
    elif args.cmd == "record":
        record_predictions(args.pred)
    else:
        grade()


if __name__ == "__main__":
    main()
