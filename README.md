- **App:** https://mlb-daily-slates.netlify.app
- **Repo:** https://github.com/StayOccupiedVR/slate-lab
- **Daily predictions:** https://StayOccupiedVR.github.io/slate-lab/predictions.json
- **Live record:** https://StayOccupiedVR.github.io/slate-lab/report.json

## What runs automatically (all ET)

| Time | What happens |
|---|---|
| 7:00 AM | Grade yesterday vs finals and closing odds · retrain · score today · file predictions to git |
| 11:00 AM | Odds snapshot (openers) |
| 6:30 PM | Odds snapshot (east-coast closings) |
| 9:30 PM | Odds snapshot (west-coast closings) |

Every prediction and odds snapshot is committed to a public repo, so git
timestamps prove predictions existed before first pitch. `record` refuses to
overwrite a filed day. That tamper-evidence is the product's credibility.

## The model

Logistic regression over two features:

1. **Pythagorean win% differential** — team strength from runs scored/allowed
2. **Starter ERA edge** — prior-regressed (50 IP), blended 75/25 with last five starts

Validated walk-forward by season. On the 2025 holdout (n=2,123):

| | log loss |
|---|---|
| coin flip | 0.6931 |
| always home at 54% | 0.6899 |
| team record only | 0.6835 |
| **shipped model** | **0.6827** |
| a perfect forecaster | ~0.673–0.680 |

The last row is the one that matters. Most MLB games are near coin flips — a
true 55% game costs a *perfect* predictor 0.688 — so the whole distance between
guessing and omniscience is about 0.015, and this model has covered a third to
two-thirds of it. Any result below ~0.67 means a data leak, not a breakthrough.

**Tested and rejected, with evidence** (see slate-lab README results log):
- Recent form (L10) — raised held-out log loss
- Rest days / back-to-backs — no signal
- Gradient boosting — could not beat logistic out of sample
- Rotation-relative starter rating — moving baseline; measured roster shape,
  not pitching (corr −0.39 with team strength)

The anti-leakage guarantee is a test, not a promise: flip any past game's
result and every earlier feature row must be byte-identical
(`tests/test_pipeline.py`).

## Odds and grading

- **Source:** The Odds API (free tier, ~180 of 500 monthly credits used).
  Direct book scraping is ToS-prohibited and geo-blocked; the aggregator is
  the legitimate route.
- **Books:** DraftKings primary (widest state coverage, deepest liquidity),
  Hard Rock Bet fallback, per game.
- **Closing line** = last snapshot before first pitch; post-pitch snapshots
  are excluded by test.
- **report.json** compares model log loss vs the devigged closing line on the
  same games, plus the disagreement record: when model and market took
  opposite sides, who was right.

Expect the market to win. The scoreboard exists to measure *by how much* and
whether the gap narrows — not to declare victory.

## The app

Newsprint design (Libre Franklin / IBM Plex Mono), three tabs:

- **Board** — one card per game: model probabilities big, tonight's starters
  with ERA, everything else (form, bullpen use, splits, confirmed lineups)
  behind a Details fold in plain English. Sort by time or by model-vs-ML
  disagreement. "Lineups in" flag when official lineups post.
- **Value** — arrives pre-filled from any game's "Check the value" button.
  Verdict first, in dollars: *has value (+$N per $100)* / *too thin to bet
  (<$5)* / *no value*. A break-even meter with a draggable estimate. The
  full math (fair prices, book's cut, half-Kelly) folded below.
- **Backtest** — replays past dates in the browser using only what was
  knowable each morning; Brier, log loss, calibration, weight fitting with a
  chronological train/test split.

Lineups come from the MLB Stats API (`hydrate=lineups`) — the same official
source Rotowire republishes — at zero extra request cost.

## Decisions made, and why

- **One sport until the gate is passed.** Every sport is a separate pipeline,
  model, and validation. MLB is the easiest to model and it isn't beaten yet.
- **Logistic over gbdt.** The trees tied, and a tie means memorized noise.
- **Files over database for the ledger.** Git history is tamper-evident;
  database rows are not.
- **DraftKings as benchmark.** Most states, most liquidity.
- **Analytics positioning, not picks.** The model sits ~0.008 behind a
  closing line that itself sits behind a ~4.5% vig. "We show the math and a
  verified live record" is honest and defensible; "we beat the book" is
  neither.

## Deliberately deferred

| Item | Why deferred | Unblocks when |
|---|---|---|
| Multi-book odds storage | Cheap, but pointless until ledger has volume | Anytime — highest-leverage next build |
| Second scoring pass at lineup-post time | Needs the two-prediction ledger design | After 60 days of baseline record |
| Lineup-strength model feature | Backtestable via boxscores; live timing gap | With the second scoring pass |
| Statcast features (xFIP, K-BB%, bullpen quality, park factors) | Each worth maybe 0.001–0.003; ablation must judge | Anytime; run `make statcast` |
| Accounts / payments | No product until the record exists | See the gate |
| Affiliate partnerships | Requires traffic | After public record + audience |

## Done since first draft

- **Multi-sport adapter refactor** (July 30) — `slate_lab/sports/` registry;
  MLB is a pure delegate to the leak-tested modules (enforced by test);
  NFL adapter slot reserved with design notes for the September build.
  All commands take `--sport`, defaulting to `mlb`; ledger keeps legacy
  MLB paths so history is uninterrupted.

## The monetization gate

No dollar changes hands until all five hold:

1. Model beats team-record-only on held-out log loss across **two** separate
   test seasons.
2. Calibration holds within noise on those seasons.
3. **60+ days of live record** — pre-pitch git-timestamped predictions,
   graded, losses included.
4. Honest positioning: analytics and education. Nothing that implies the
   model out-predicts the closing line, because it doesn't.
5. A real attorney has reviewed the offering. Betting-adjacent products touch
   state gambling regulation, advertising rules, and consumer protection.

## Operating notes

- **Costs:** $0/month. GitHub Actions free tier, Netlify free tier, MLB API
  free, The Odds API free tier.
- **Keys:** `ODDS_API_KEY` lives only in GitHub Actions secrets. Never in code.
- **Updating the app:** Netlify dashboard → mlb-daily-slates → Deploys → drag
  the folder. Drops to netlify.com/drop create a *new* site — don't.
- **Updating the model/pipeline:** edit, `git push` to slate-lab; next
  scheduled run uses it.
- **Reading the record:** ~100 graded games after a week (noise), ~400 after
  a month (worth reading). Disagreement stats need several hundred
  disagreements, i.e. months.

## Current status

Backtest phase complete. Live ledger at day zero — the record accumulates
from the first scheduled run. Next scheduled decision point: review
report.json at ~200 graded games.
