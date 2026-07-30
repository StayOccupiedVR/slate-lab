"""NFL adapter — the September build slot. Interface real, internals pending.

Design notes committed now so the build starts from decisions, not a blank
file:

DATA    nfl_data_py (nflverse): free schedules, results, play-by-play with
        EPA. Better raw material than the MLB feed.
TEAM    Pythagorean on point differential, exponent ~2.37. Regress harder
        than MLB — 17 games is a tiny sample, prior season must blend in
        early-season (weight prior ~0.6 in week 1, decay to ~0 by week 8).
QB      The starting-pitcher analog. QB out/in is the single largest line
        mover (3-7 pts). Feature: starter continuity flag + QB EPA/play,
        prior-regressed. Confirmed starters are knowable pre-kickoff.
REST    Unlike MLB, rest matters here: bye weeks, Thursday short weeks,
        London trips. Cheap features, let the ablation judge.
HFA     ~0.145 logit (~2.2 points), drifting down league-wide for years —
        fit it, don't hardcode.
VALIDATE  272 games/season. Walk-forward by season still, but expect wide
        error bars; two seasons of holdout minimum before trusting anything.
ODDS    The Odds API: americanfootball_nfl. Same books, same ledger.
MARKET  Sharpest closing line in US sports. The gap will be larger than
        MLB's. The product claim is the verified record, not the edge.
"""
from __future__ import annotations

from types import SimpleNamespace

_MSG = ("NFL adapter is scheduled for the September build. "
        "The interface is reserved; see this file's design notes.")


def _todo(*_a, **_k):
    raise SystemExit(_MSG)


ADAPTER = SimpleNamespace(
    key="nfl",
    name="NFL",
    odds_sport="americanfootball_nfl",
    books=["draftkings", "hardrockbet"],
    data_prefix="nfl",
    team_name_to_id={},                 # filled with the ingest build
    season_bounds=lambda season: (f"{season}-09-01", f"{season+1}-02-15"),
    ingest=_todo,
    build_features=_todo,
    GROUPS={},
    label="away_won",
    slate=_todo,
    slate_features=_todo,
    baseline=None,
)
