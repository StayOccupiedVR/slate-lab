"""NFL weekly ledger — file Wednesday, revise only for QB news, grade with CLV.

FORMAT  One JSON per week: data/nfl/weeks/{season}_w{NN}.json
        Entries are append-only. A filed probability is never edited;
        a Sunday QB-change re-file lands in `revised` beside the
        original. `close` and `result` are written by the grader after
        the fact and never touch the filed numbers.

FILING  Wednesday of each week: every game that week, p_away from the
        validated model (train on all completed games to date), the
        best line available at filing time, and the QB assumption used
        (incumbent = each team's last starter). nflverse leaves QB names
        blank until games are played, so incumbency is the honest
        Wednesday assumption — the revision path exists precisely for
        when that assumption breaks.

REVISE  Before kickoff only, and only when the starting QB differs from
        the filed assumption. Both numbers stay on the record.

GRADE   After results land: closing line = latest snapshot at or before
        kickoff. Report carries log loss (ours vs devigged close),
        cover-relevant fields for the future margin model, and CLV —
        did the market move toward our side after we filed? CLV
        accumulates signal every game, which matters on a 272-game
        season where win/loss noise dominates until midseason.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

DATA = Path("data/nfl")
MODEL_TAG = "nfl-v1"


# ---------------------------------------------------------------- helpers
def _ml_to_prob(ml):
    if ml is None or (isinstance(ml, float) and np.isnan(ml)):
        return None
    ml = float(ml)
    return (-ml) / ((-ml) + 100) if ml < 0 else 100 / (ml + 100)


def _devig_away(ml_away, ml_home):
    a, h = _ml_to_prob(ml_away), _ml_to_prob(ml_home)
    if a is None or h is None:
        return None
    return a / (a + h)


def _week_path(season: int, week: int) -> Path:
    return DATA / "weeks" / f"{season}_w{week:02d}.json"


def _load_week(season: int, week: int) -> dict:
    p = _week_path(season, week)
    if p.exists():
        return json.loads(p.read_text())
    return {"season": season, "week": week, "games": []}


def _save_week(doc: dict) -> None:
    p = _week_path(doc["season"], doc["week"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1))


def _snapshots() -> list[tuple[datetime, dict]]:
    out = []
    d = DATA / "odds"
    if not d.exists():
        return out
    for f in sorted(d.glob("*.json")):
        try:
            doc = json.loads(f.read_text())
            ts = datetime.strptime(
                doc["captured"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            out.append((ts, doc))
        except Exception:
            continue
    return out


def _line_for(snap: dict, away: str, home: str):
    for ev in snap.get("events", []):
        if ev.get("away_id") == away and ev.get("home_id") == home:
            books = ev.get("books", {})
            for bk in ("draftkings", "fanduel", "hardrockbet"):
                if bk in books:
                    b = books[bk]
                    return {"book": bk, "ml_away": b["ml_away"],
                            "ml_home": b["ml_home"]}
    return None


def _latest_line(away: str, home: str, before: datetime | None = None):
    best = None
    for ts, snap in _snapshots():
        if before is not None and ts > before:
            continue
        line = _line_for(snap, away, home)
        if line is not None:
            best = {**line, "captured": ts.strftime("%Y%m%dT%H%M%SZ")}
    return best


# ---------------------------------------------------------------- model
def _fit(sp, con):
    from sklearn.linear_model import LinearRegression
    df = sp.build_features(con)
    cols = [c for g in sp.GROUPS.values() for c in g]
    m = LogisticRegression(C=1.0).fit(df[cols], df[sp.label])
    # margin: home - away, matching nflverse's `result` convention
    mm = LinearRegression().fit(df[cols], -df["margin"])
    return m, mm, cols


def _week_rows(sp, con, season: int, week: int) -> pd.DataFrame:
    """Point-in-time features for the SCHEDULED games of one week.

    Ratings come from this season's played games (same fold pattern as
    build_features); QB flags compare the schedule's announced starter,
    when present, to each team's last starter — blank means incumbent.
    """
    from collections import defaultdict
    nfl = __import__("slate_lab.sports.nfl", fromlist=["PYTH_EXP"])
    played = pd.read_sql(
        "SELECT * FROM nfl_games WHERE season=? AND game_type='REG' "
        "AND away_score IS NOT NULL ORDER BY week, gameday", con,
        params=(season,))
    sched = pd.read_sql(
        "SELECT * FROM nfl_games WHERE season=? AND week=? AND "
        "game_type='REG' AND away_score IS NULL ORDER BY gameday", con,
        params=(season, week))
    pf = defaultdict(float); pa = defaultdict(float)
    gp = defaultdict(int); qb_last: dict = {}
    for r in played.itertuples():
        pf[r.away_team] += r.away_score; pa[r.away_team] += r.home_score
        gp[r.away_team] += 1
        pf[r.home_team] += r.home_score; pa[r.home_team] += r.away_score
        gp[r.home_team] += 1
        qb_last[r.away_team] = r.away_qb; qb_last[r.home_team] = r.home_qb

    def pyth(t):
        f = pf[t] + nfl.PRIOR_PPG * nfl.PRIOR_GAMES
        a = pa[t] + nfl.PRIOR_PPG * nfl.PRIOR_GAMES
        return f ** nfl.PYTH_EXP / (f ** nfl.PYTH_EXP + a ** nfl.PYTH_EXP)

    rows = []
    for r in sched.itertuples():
        # sqlite NULLs surface as NaN once any row holds a string; NaN is
        # truthy, so only genuine strings count as an announced starter.
        qa = r.away_qb if isinstance(r.away_qb, str) else None
        qh = r.home_qb if isinstance(r.home_qb, str) else None
        rows.append({
            "game_id": r.game_id, "gameday": r.gameday,
            "away": r.away_team, "home": r.home_team,
            "pyth_diff": pyth(r.away_team) - pyth(r.home_team),
            "rest_diff": float((r.away_rest or 7) - (r.home_rest or 7)),
            "qb_new_away": int(qa is not None and qb_last.get(r.away_team)
                               is not None and qa != qb_last[r.away_team]),
            "qb_new_home": int(qh is not None and qb_last.get(r.home_team)
                               is not None and qh != qb_last[r.home_team]),
            "div": int(r.div_game or 0),
            "qb_assumed_away": qa or qb_last.get(r.away_team),
            "qb_assumed_home": qh or qb_last.get(r.home_team),
        })
    return pd.DataFrame(rows)


def upcoming_week(con, lookahead_days: int = 10) -> tuple[int, int] | None:
    """(season, week) of the next unplayed REG game within the lookahead
    window, or None — which is how every cron knows to sleep through the
    preseason and the offseason without configuration."""
    from datetime import date, timedelta
    today = date.today()
    horizon = (today + timedelta(days=lookahead_days)).isoformat()
    row = pd.read_sql(
        "SELECT season, week FROM nfl_games WHERE game_type='REG' AND "
        "away_score IS NULL AND gameday >= ? AND gameday <= ? "
        "ORDER BY gameday LIMIT 1", con,
        params=(today.isoformat(), horizon))
    if row.empty:
        return None
    return int(row.season.iloc[0]), int(row.week.iloc[0])


# ---------------------------------------------------------------- commands
def file_week(sp, con, season: int, week: int) -> None:
    """File every not-yet-filed game of the week. Idempotent: existing
    entries are never touched, so re-running cannot alter the record."""
    doc = _load_week(season, week)
    have = {g["game_id"] for g in doc["games"]}
    wk = _week_rows(sp, con, season, week)
    if wk.empty:
        print(f"week {week}: no scheduled games to file")
        return
    model, margin_model, cols = _fit(sp, con)
    probs = model.predict_proba(wk[cols])[:, 1]
    margins = margin_model.predict(wk[cols])
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    added = 0
    for row, p, mg in zip(wk.itertuples(), probs, margins):
        if row.game_id in have:
            continue
        doc["games"].append({
            "game_id": row.game_id, "gameday": row.gameday,
            "away": row.away, "home": row.home,
            "filed": now, "model": MODEL_TAG,
            "p_away": round(float(p), 4),
            "proj_home_margin": round(float(mg), 1),
            "qb_assumed": {"away": row.qb_assumed_away,
                           "home": row.qb_assumed_home},
            "line_at_file": _latest_line(row.away, row.home),
            "revised": None, "close": None, "result": None,
        })
        added += 1
    _save_week(doc)
    print(f"week {week}: filed {added} new "
          f"({len(doc['games'])} total on record)")


def revise_week(sp, con, season: int, week: int) -> None:
    """Re-file ONLY games whose starting QB now differs from the filed
    assumption, and only before their game day. Original stays."""
    doc = _load_week(season, week)
    if not doc["games"]:
        print(f"week {week}: nothing filed yet")
        return
    wk = _week_rows(sp, con, season, week)
    if wk.empty:
        print(f"week {week}: no unplayed games left to revise")
        return
    model, margin_model, cols = _fit(sp, con)
    probs = {r.game_id: p for r, p in
             zip(wk.itertuples(), model.predict_proba(wk[cols])[:, 1])}
    now_d = datetime.now(timezone.utc).date().isoformat()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    n = 0
    by_id = {r.game_id: r for r in wk.itertuples()}
    for g in doc["games"]:
        r = by_id.get(g["game_id"])
        if r is None or g["gameday"] < now_d or g["revised"] is not None:
            continue
        qa_changed = (r.qb_assumed_away or "") != (g["qb_assumed"]["away"] or "")
        qh_changed = (r.qb_assumed_home or "") != (g["qb_assumed"]["home"] or "")
        if qa_changed or qh_changed:
            g["revised"] = {
                "filed": now, "reason": "qb_change",
                "p_away": round(float(probs[g["game_id"]]), 4),
                "qb": {"away": r.qb_assumed_away, "home": r.qb_assumed_home},
            }
            n += 1
    _save_week(doc)
    print(f"week {week}: {n} QB revision(s) filed")


def grade(sp, con, season: int) -> None:
    """Fill close + result for finished games; write data/nfl/report.json."""
    results = pd.read_sql(
        "SELECT game_id, away_score, home_score, gameday FROM nfl_games "
        "WHERE season=? AND away_score IS NOT NULL", con, params=(season,))
    res = {r.game_id: r for r in results.itertuples()}
    stats = []
    for wf in sorted((DATA / "weeks").glob(f"{season}_w*.json")) \
            if (DATA / "weeks").exists() else []:
        doc = json.loads(wf.read_text())
        changed = False
        for g in doc["games"]:
            r = res.get(g["game_id"])
            if r is None:
                continue
            if g["result"] is None:
                g["result"] = {"away_score": float(r.away_score),
                               "home_score": float(r.home_score)}
                changed = True
            if g["close"] is None:
                kickoff = datetime.fromisoformat(
                    g["gameday"]).replace(hour=23, minute=59,
                                          tzinfo=timezone.utc)
                g["close"] = _latest_line(g["away"], g["home"],
                                          before=kickoff)
                changed = True
            stats.append(g)
        if changed:
            _save_week(doc)

    graded = [g for g in stats if g["result"] is not None]
    scored = []
    for g in graded:
        p = (g["revised"] or {}).get("p_away", g["p_away"])
        won = int(g["result"]["away_score"] > g["result"]["home_score"])
        entry = {"p": p, "won": won, "ll": -np.log(p if won else 1 - p)}
        close_p = (_devig_away(g["close"]["ml_away"], g["close"]["ml_home"])
                   if g["close"] else None)
        file_p = (_devig_away(g["line_at_file"]["ml_away"],
                              g["line_at_file"]["ml_home"])
                  if g["line_at_file"] else None)
        actual_hm = g["result"]["home_score"] - g["result"]["away_score"]
        if g.get("proj_home_margin") is not None:
            entry["margin_ae"] = abs(g["proj_home_margin"] - actual_hm)
        if g.get("close") and g["close"].get("spread") is not None:
            entry["market_margin_ae"] = abs(g["close"]["spread"] - actual_hm)
        if close_p is not None:
            entry["market_ll"] = -np.log(close_p if won else 1 - close_p)
            if file_p is not None:
                # CLV: movement of the market toward our side after filing
                our_side_away = p >= 0.5
                mv = close_p - file_p
                entry["clv"] = mv if our_side_away else -mv
        scored.append(entry)

    if scored:
        report = {
            "season": season, "n": len(scored),
            "logloss": round(float(np.mean([s["ll"] for s in scored])), 4),
            "market_logloss": round(float(np.mean(
                [s["market_ll"] for s in scored if "market_ll" in s])), 4)
            if any("market_ll" in s for s in scored) else None,
            "clv_avg_pts": round(100 * float(np.mean(
                [s["clv"] for s in scored if "clv" in s])), 2)
            if any("clv" in s for s in scored) else None,
            "margin_mae": round(float(np.mean(
                [s["margin_ae"] for s in scored if "margin_ae" in s])), 2)
            if any("margin_ae" in s for s in scored) else None,
            "market_margin_mae": round(float(np.mean(
                [s["market_margin_ae"] for s in scored
                 if "market_margin_ae" in s])), 2)
            if any("market_margin_ae" in s for s in scored) else None,
        }
        (DATA / "report.json").write_text(json.dumps(report, indent=1))
        print(f"graded {report['n']} games | logloss {report['logloss']}"
              f" | market {report['market_logloss']}"
              f" | CLV {report['clv_avg_pts']} pts")
    else:
        print("nothing gradeable yet")
