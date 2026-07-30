"""Offline tests for the ledger: devig math, book preference, closing-snapshot
selection, grading metrics. No network required."""
import json, sys, math, datetime as dt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import tempfile, os

from slate_lab import ledger


def test_devig_math():
    # -150 / +130: raw 60.0% + 43.5% = 103.5%; fair away = .5799
    pa, ph = ledger.devig(ledger.ml_to_prob(-150), ledger.ml_to_prob(130))
    assert abs(pa + ph - 1.0) < 1e-12
    assert abs(pa - 0.5799) < 0.001, pa
    print(f"  devig: -150/+130 -> fair away {pa:.4f} ✓")


def test_book_preference_and_closing_selection(tmpdir):
    ledger.DATA = Path(tmpdir) / "data"
    odir = ledger.DATA / "odds"; odir.mkdir(parents=True)
    ev = {"commence": "2026-07-28T23:10:00Z", "away_id": 147, "home_id": 145,
          "away": "New York Yankees", "home": "Chicago White Sox"}
    # early snapshot: only DK. later snapshot (still pre-pitch): HR + DK.
    # post-pitch snapshot: must be ignored.
    snaps = [
      ("20260728T150000Z", {"draftkings": {"ml_away": -140, "ml_home": 120}}),
      ("20260728T223000Z", {"hardrockbet": {"ml_away": -150, "ml_home": 130},
                            "draftkings":  {"ml_away": -145, "ml_home": 125}}),
      ("20260729T010000Z", {"hardrockbet": {"ml_away": -200, "ml_home": 170}}),
    ]
    for ts, books in snaps:
        (odir / f"{ts}.json").write_text(json.dumps(
            {"captured": ts, "events": [dict(ev, books=books)]}))
    close = ledger._closing_odds()
    row = close[("2026-07-28", 147, 145)]
    assert row["book"] == "draftkings", row
    assert row["ml_away"] == -145, "should use last PRE-pitch snapshot, not post"
    print(f"  closing: picked draftkings -145 from 22:30Z snapshot ✓")


def test_grading_metrics(tmpdir):
    rows = [
        {"away_won": True,  "p_model": 0.60, "p_market": 0.55},
        {"away_won": False, "p_model": 0.45, "p_market": 0.52},
        {"away_won": True,  "p_model": 0.40, "p_market": 0.48},
    ]
    n = len(rows)
    ll = sum(ledger._ll(r["away_won"], r["p_model"]) for r in rows) / n
    hand = -(math.log(0.60) + math.log(0.55) + math.log(0.40)) / 3
    assert abs(ll - hand) < 1e-12
    print(f"  grading: logloss matches hand calc ({ll:.4f}) ✓")


if __name__ == "__main__":
    import tempfile
    test_devig_math()
    with tempfile.TemporaryDirectory() as td:
        test_book_preference_and_closing_selection(td)
    with tempfile.TemporaryDirectory() as td:
        test_grading_metrics(td)
    print("\nLEDGER TESTS PASSED")
