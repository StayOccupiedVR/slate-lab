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
    assert str(ledger.DATA) == "data/nfl"
    ledger._configure("mlb")   # restore default
    print("  ledger: per-sport odds URL and data paths \u2713")



def test_nfl_adapter_offline():
    """NFL adapter: synthetic ingest-free test of features + trainability.

    Builds a fake nfl_games table (no network), checks the feature frame is
    point-in-time sane, market columns stay out of the model inputs, and the
    shared trainer runs end to end on it.
    """
    import sqlite3
    import random
    from slate_lab.sports import get_sport

    sp = get_sport("nfl")
    con = sqlite3.connect(":memory:")
    con.executescript(__import__("slate_lab.sports.nfl", fromlist=["SCHEMA"]).SCHEMA)
    rng = random.Random(7)
    teams = [f"T{i}" for i in range(16)]
    strength = {t: rng.gauss(0, 4) for t in teams}
    rows = []
    for season in (2020, 2021, 2022, 2023):
        for week in range(1, 19):
            order = rng.sample(teams, len(teams))
            for i in range(0, len(order), 2):
                a, h = order[i], order[i + 1]
                ma = 21 + strength[a] - 0.5 * strength[h]
                mh = 23 + strength[h] - 0.5 * strength[a]   # ~2pt HFA
                asc = max(0, round(rng.gauss(ma, 9)))
                hsc = max(0, round(rng.gauss(mh, 9)))
                rows.append((f"{season}_{week}_{a}_{h}", season, week,
                             f"{season}-10-{week:02d}", a, h, asc, hsc,
                             7, 7, f"QB{a}", f"QB{h}", 0,
                             None, None, -120, 100, "REG"))
    con.executemany(
        "INSERT INTO nfl_games VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows)
    con.commit()

    df = sp.build_features(con)
    assert len(df) > 400, "too few scoreable synthetic games"
    assert df.away_won.isin((0, 1)).all()
    # market/eval columns present but NOT in any model group
    model_cols = {c for cols in sp.GROUPS.values() for c in cols}
    for c in ("market_p_away", "close_spread", "margin"):
        assert c in df.columns and c not in model_cols
    # early-season games excluded until both teams have MIN_GP
    assert df.week.min() > 1

    from slate_lab.train import walk_forward
    res = walk_forward(df, 2023, list(sp.GROUPS), sp)
    ll = res["logistic"]["logloss"]
    assert 0.5 < ll < 0.72, f"synthetic logloss out of range: {ll}"
    print(f"  nfl synthetic: {len(df)} games, holdout logloss {ll:.4f}")


def test_nba():
    """NBA features on a synthetic season: point-in-time, b2b, rest."""
    import sqlite3
    from slate_lab.sports import get_sport
    sp = get_sport("nba")
    con = sqlite3.connect(":memory:")
    con.executescript(__import__("slate_lab.sports.nba",
                                 fromlist=["SCHEMA"]).SCHEMA)
    import random
    rng = random.Random(3)
    teams = [str(100 + i) for i in range(8)]
    gid = 0
    import datetime as dt
    d = dt.date(2022, 10, 20)
    rows = []
    for night in range(70):
        pairs = rng.sample(teams, 6)
        for i in range(0, 6, 2):
            a, h = pairs[i], pairs[i + 1]
            good = int(a) % 3 == 0
            asc = rng.gauss(114 if good else 108, 8)
            hsc = rng.gauss(112, 8)
            rows.append((f"g{gid}", 2023, d.isoformat(), a, h,
                         f"T{a}", f"T{h}", asc, hsc, 1, "REG"))
            gid += 1
        d += dt.timedelta(days=1)
    con.executemany(
        "INSERT OR REPLACE INTO nba_games VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        rows)
    con.commit()
    df = sp.build_features(con)
    assert len(df) > 60, "burn-in leaves scoreable games"
    assert set(sp.GROUPS["team"] + sp.GROUPS["schedule"]) <= set(df.columns)
    assert df.b2b_away.isin((0, 1)).all() and df.b2b_away.sum() > 0, \
        "back-to-backs detected in a dense schedule"
    assert df.pyth_diff.abs().max() < 0.5
    # point-in-time: features for an early game can't see later games
    early = df.sort_values("date").iloc[0]
    assert abs(early.pyth_diff) < 0.25, "burn-in-era pyth stays shrunk"
    print(f"  nba: {len(df)} feature games, b2b rate "
          f"{df.b2b_away.mean():.0%}, features sane")
    print("NBA ADAPTER TEST PASSED")




def test_nhl():
    """NHL features on a dense synthetic schedule: b2b-heavy, sane."""
    import sqlite3
    import random
    import datetime as dt
    from slate_lab.sports import get_sport
    sp = get_sport("nhl")
    con = sqlite3.connect(":memory:")
    con.executescript(__import__("slate_lab.sports.nhl",
                                 fromlist=["SCHEMA"]).SCHEMA)
    rng = random.Random(6)
    teams = [str(200 + i) for i in range(8)]
    gid = 0
    d = dt.date(2022, 10, 12)
    rows = []
    for night in range(80):
        pairs = rng.sample(teams, 6)
        for i in range(0, 6, 2):
            a, h = pairs[i], pairs[i + 1]
            asc = max(0, round(rng.gauss(3.2 if int(a) % 2 else 2.8, 1.6)))
            hsc = max(0, round(rng.gauss(3.0, 1.6)))
            rows.append((f"h{gid}", 2023, d.isoformat(), a, h,
                         f"T{a}", f"T{h}", asc, hsc, 1, "REG"))
            gid += 1
        d += dt.timedelta(days=1)
    con.executemany(
        "INSERT OR REPLACE INTO nhl_games VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        rows)
    con.commit()
    df = sp.build_features(con)
    assert len(df) > 80
    assert df.b2b_away.sum() > 0 and df.b2b_home.sum() > 0, \
        "dense schedule must produce back-to-backs"
    assert (df.rest_diff != 0).sum() > 0, "rest differential varies"
    assert df.pyth_diff.abs().max() < 0.5
    print(f"  nhl: {len(df)} feature games, b2b_away rate "
          f"{df.b2b_away.mean():.0%}, rest varies")
    print("NHL ADAPTER TEST PASSED")


if __name__ == "__main__":
    test_contract()
    test_mlb_is_a_delegate()
    test_ledger_configures_per_sport()
    test_nfl_adapter_offline()
    test_nba()
    test_nhl()
    print("\nSPORTS TESTS PASSED")
