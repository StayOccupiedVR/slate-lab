"""MLB adapter — a thin shell around the proven, leak-tested modules.

Deliberately contains no logic. ingest.py and features.py carry the
tripwire-verified point-in-time guarantees; this file only routes to them,
so the guarantees travel with the code instead of being re-implemented.
"""
from __future__ import annotations

from types import SimpleNamespace

from .. import features as F
from .. import ingest as I
from ..models import baseline_predict


def _slate(date: str):
    from ..score import todays_slate
    return todays_slate(date)


def _slate_features(con, slate, date):
    from ..score import slate_features
    return slate_features(con, slate, date)


ADAPTER = SimpleNamespace(
    key="mlb",
    name="MLB",
    odds_sport="baseball_mlb",
    books=["draftkings", "fanduel", "hardrockbet"],
    data_prefix="",                     # legacy paths; keeps existing ledger history
    team_name_to_id=None,               # ledger falls back to its MLB map
    season_bounds=lambda season: (f"{season}-01-01", f"{season}-12-31"),
    ingest=I.ingest_season,
    build_features=F.build_features,
    GROUPS=F.GROUPS,
    label="away_won",
    slate=_slate,
    slate_features=_slate_features,
    baseline=baseline_predict,
)
