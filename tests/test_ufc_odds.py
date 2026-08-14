"""UFC odds capture: book mapping and de-vigged consensus."""
import sys
sys.path.insert(0, ".")
from slate_lab.capture_ufc_odds import build_doc, implied


def test_implied():
    assert abs(implied(-450) - 450/550) < 1e-9
    assert abs(implied(340) - 100/440) < 1e-9
    print("  implied odds math correct")


def test_build_doc():
    events = [{
        "home_team": "Fighter One", "away_team": "Fighter Two",
        "commence_time": "2026-08-15T22:00:00Z",
        "bookmakers": [
            {"key": "draftkings", "markets": [{"key": "h2h", "outcomes": [
                {"name": "Fighter One", "price": -450},
                {"name": "Fighter Two", "price": 340}]}]},
            {"key": "fanduel", "markets": [{"key": "h2h", "outcomes": [
                {"name": "Fighter Two", "price": 330},
                {"name": "Fighter One", "price": -430}]}]},
            {"key": "bovada", "markets": [{"key": "h2h", "outcomes": [
                {"name": "Fighter One", "price": -500},
                {"name": "Fighter Two", "price": 400}]}]},
        ],
    }, {
        "home_team": "No Books Guy", "away_team": "Other Guy",
        "bookmakers": [],
    }]
    doc = build_doc(events)
    assert len(doc["fights"]) == 1, "bookless fights dropped"
    f = doc["fights"][0]
    assert set(f["books"]) == {"draftkings", "fanduel"}, "only our books"
    assert f["books"]["draftkings"] == {"a": -450, "b": 340}
    assert f["books"]["fanduel"] == {"a": -430, "b": 330}, "outcome order-proof"
    assert abs(f["market"]["a"] + f["market"]["b"] - 1.0) < 1e-9, "de-vigged"
    assert 0.75 <= f["market"]["a"] <= 0.84, f["market"]
    assert "captured" in doc
    print(f"  consensus {f['market']['a']:.1%} favorite, vig removed")


if __name__ == "__main__":
    test_implied()
    test_build_doc()
    print("\nUFC ODDS TESTS PASSED")
