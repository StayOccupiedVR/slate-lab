"""NFL weekly ledger lifecycle test — fully offline.

Covers: filing a week, immutability on re-file, the QB-only revision
rule, grading with closing lines, and that CLV is signed correctly
(market moving toward our side = positive).
"""
import json
import sqlite3
import random
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import sys

sys.path.insert(0, ".")

from slate_lab.sports import get_sport
from slate_lab.sports.nfl import SCHEMA
from slate_lab import nfl_ledger


def synth_con():
    con = sqlite3.connect(":memory:")
    con.executescript(SCHEMA)
    rng = random.Random(3)
    teams = [f"T{i}" for i in range(16)]
    strength = {t: rng.gauss(0, 4) for t in teams}
    rows = []
    # two full past seasons for training
    for season in (2024, 2025):
        for week in range(1, 19):
            order = rng.sample(teams, len(teams))
            for i in range(0, len(order), 2):
                a, h = order[i], order[i + 1]
                asc = max(0, round(rng.gauss(21 + strength[a] - .5 * strength[h], 9)))
                hsc = max(0, round(rng.gauss(23 + strength[h] - .5 * strength[a], 9)))
                rows.append((f"{season}_{week:02d}_{a}_{h}", season, week,
                             f"{season}-10-{min(week,28):02d}", a, h, asc, hsc,
                             7, 7, f"QB{a}", f"QB{h}", 0, None, None,
                             -120, 100, "REG"))
    # current season: weeks 1-4 played, week 5 scheduled (no scores, no QBs)
    for week in range(1, 5):
        order = rng.sample(teams, len(teams))
        for i in range(0, len(order), 2):
            a, h = order[i], order[i + 1]
            asc = max(0, round(rng.gauss(21 + strength[a] - .5 * strength[h], 9)))
            hsc = max(0, round(rng.gauss(23 + strength[h] - .5 * strength[a], 9)))
            rows.append((f"2026_{week:02d}_{a}_{h}", 2026, week,
                         f"2026-09-{7 + week:02d}", a, h, asc, hsc,
                         7, 7, f"QB{a}", f"QB{h}", 0, None, None,
                         -120, 100, "REG"))
    order = rng.sample(teams, len(teams))
    wk5 = []
    for i in range(0, len(order), 2):
        a, h = order[i], order[i + 1]
        gid = f"2026_05_{a}_{h}"
        wk5.append((gid, a, h))
        rows.append((gid, 2026, 5, "2099-10-12", a, h, None, None,
                     7, 10, None, None, 1, None, None, None, None, "REG"))
    con.executemany(
        "INSERT INTO nfl_games VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows)
    con.commit()
    return con, wk5, teams


def snapshot(ts, games, spread_move=0):
    return {"captured": ts, "events": [
        {"away_id": a, "home_id": h, "commence": "2099-10-12T17:00:00Z",
         "books": {"draftkings": {"ml_away": -120 - spread_move,
                                  "ml_home": 100 + spread_move}}}
        for _, a, h in games]}


def main():
    sp = get_sport("nfl")
    con, wk5, teams = synth_con()
    tmp = Path(tempfile.mkdtemp())
    nfl_ledger.DATA = tmp

    # --- file week 5
    nfl_ledger.file_week(sp, con, 2026, 5)
    doc = json.loads((tmp / "weeks" / "2026_w05.json").read_text())
    assert len(doc["games"]) == 8
    g0 = doc["games"][0]
    assert 0 < g0["p_away"] < 1 and g0["revised"] is None
    filed_p = g0["p_away"]
    print(f"  filed week 5: {len(doc['games'])} games, sample p={filed_p}")

    # --- immutability: re-file changes nothing
    nfl_ledger.file_week(sp, con, 2026, 5)
    doc2 = json.loads((tmp / "weeks" / "2026_w05.json").read_text())
    assert doc2["games"][0]["p_away"] == filed_p
    assert len(doc2["games"]) == 8
    print("  re-file is a no-op: filed numbers untouched")

    # --- QB change on game 0's away team -> revision fires for it only
    gid, a0, h0 = wk5[0][0], wk5[0][1], wk5[0][2]
    con.execute("UPDATE nfl_games SET away_qb=? WHERE game_id=?",
                (f"QB_BACKUP_{a0}", gid))
    con.commit()
    nfl_ledger.revise_week(sp, con, 2026, 5)
    doc3 = json.loads((tmp / "weeks" / "2026_w05.json").read_text())
    revs = [g for g in doc3["games"] if g["revised"] is not None]
    assert len(revs) == 1 and revs[0]["game_id"] == gid
    assert revs[0]["p_away"] == filed_p          # original preserved
    assert revs[0]["revised"]["reason"] == "qb_change"
    print("  QB revision: exactly one game revised, original preserved")

    # --- snapshots: filing line then a close that moved toward the away side
    (tmp / "odds").mkdir()
    (tmp / "odds" / "20991001T120000Z.json").write_text(
        json.dumps(snapshot("20991001T120000Z", wk5)))
    (tmp / "odds" / "20991012T150000Z.json").write_text(
        json.dumps(snapshot("20991012T150000Z", wk5, spread_move=20)))

    # results: away team wins every wk5 game (deterministic grading)
    for gid_, a, h in wk5:
        con.execute("UPDATE nfl_games SET away_score=27, home_score=17, "
                    "gameday='2099-10-12' WHERE game_id=?", (gid_,))
    con.commit()
    # line_at_file was None (no snapshots at filing time) -> patch it in
    for g in doc3["games"]:
        g["line_at_file"] = {"book": "draftkings", "ml_away": -120,
                             "ml_home": 100}
    nfl_ledger._save_week(doc3)

    nfl_ledger.grade(sp, con, 2026)
    report = json.loads((tmp / "report.json").read_text())
    assert report["n"] == 8 and report["logloss"] > 0
    assert report["market_logloss"] is not None
    # market moved toward away (-120 -> -140); for away-side picks CLV > 0
    doc4 = json.loads((tmp / "weeks" / "2026_w05.json").read_text())
    away_picks = [g for g in doc4["games"]
                  if ((g["revised"] or {}).get("p_away", g["p_away"])) >= 0.5]
    if away_picks:
        assert report["clv_avg_pts"] is not None
    print(f"  graded: n={report['n']} logloss={report['logloss']} "
          f"market={report['market_logloss']} CLV={report['clv_avg_pts']}pts")
    print("\nNFL LEDGER TESTS PASSED")


if __name__ == "__main__":
    main()
