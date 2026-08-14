"""Sport adapter registry.

One sport = one adapter. The generic layers (train, score, ledger, models)
never import a sport directly; they ask the registry. Adding a sport means
adding a file here that fills in the interface — nothing else changes.

Contract every adapter must satisfy:

  key            short id ("mlb", "nfl")
  name           display name
  odds_sport     The Odds API sport path ("baseball_mlb")
  books          bookmaker preference order for closing lines
  data_prefix    subfolder under data/ for the ledger ("" keeps legacy paths)
  team_name_to_id  odds-feed team names -> internal ids
  season_bounds(season) -> (start_iso, end_iso) for ingest/schedule pulls
  ingest(con, season)    fill games (+ any sport tables), point-in-time safe
  build_features(con)    feature frame with an `away_won`-style label
  GROUPS                 feature-group dict for the ablation harness
  label                  name of the outcome column
  slate(date) -> [ {gamePk/id, away, home, away_id, home_id, ...} ]
  slate_features(con, slate, date) -> frame scoreable by the trained model
  baseline(df) -> np.ndarray | None   closed-form comparison model, if any
"""
from __future__ import annotations

from importlib import import_module

_REGISTRY = {"mlb": "slate_lab.sports.mlb", "nfl": "slate_lab.sports.nfl",
             "nba": "slate_lab.sports.nba"}
_LOADED: dict[str, object] = {}


def get_sport(key: str = "mlb"):
    key = (key or "mlb").lower()
    if key not in _REGISTRY:
        raise SystemExit(
            f"Unknown sport '{key}'. Available: {', '.join(sorted(_REGISTRY))}")
    if key not in _LOADED:
        _LOADED[key] = import_module(_REGISTRY[key]).ADAPTER
    return _LOADED[key]


def available() -> list[str]:
    return sorted(_REGISTRY)
