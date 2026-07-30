"""Adapter contract test: every registered sport must expose the full
interface, and the MLB adapter must be a pure delegate to the proven modules
(same functions, not copies)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slate_lab.sports import available, get_sport
from slate_lab import features as F, ingest as I

REQUIRED = ["key", "name", "odds_sport", "books", "data_prefix",
            "season_bounds", "ingest", "build_features", "GROUPS",
            "label", "slate", "slate_features", "baseline"]

def test_contract():
    for key in available():
        sp = get_sport(key)
        missing = [a for a in REQUIRED if not hasattr(sp, a)]
        assert not missing, f"{key} missing {missing}"
        assert sp.key == key
        assert isinstance(sp.books, list) and sp.books
        s, e = sp.season_bounds(2025)
        assert s < e
    print(f"  contract: {', '.join(available())} expose all "
          f"{len(REQUIRED)} attributes \u2713")

def test_mlb_is_a_delegate():
    sp = get_sport("mlb")
    assert sp.build_features is F.build_features, "mlb must reuse features.py"
    assert sp.ingest is I.ingest_season, "mlb must reuse ingest.py"
    assert sp.GROUPS is F.GROUPS
    print("  mlb: delegates to the leak-tested modules, no copies \u2713")

def test_ledger_configures_per_sport():
    from slate_lab import ledger
    sp = ledger._configure("mlb")
    assert "baseball_mlb" in ledger.ODDS_API
    assert str(ledger.DATA) == "data", "mlb must keep legacy ledger paths"
    sp = ledger._configure("nfl")
    assert "americanfootball_nfl" in ledger.ODDS_API
    assert ledger.DATA.as_posix() == "data/nfl"
    ledger._configure("mlb")   # restore default
    print("  ledger: per-sport odds URL and data paths \u2713")

if __name__ == "__main__":
    test_contract()
    test_mlb_is_a_delegate()
    test_ledger_configures_per_sport()
    print("\nSPORTS TESTS PASSED")

