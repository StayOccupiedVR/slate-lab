# slate-lab

The machine-learning half of Slate: a point-in-time backtesting pipeline,
Statcast features, a nightly scoring job, and the gate monetization has to
pass through.

Everything here obeys one rule: **a game on date D may only see data from
before D.** The test suite enforces it with a tampering tripwire — flip a
late-season result and every earlier feature row must be byte-identical.
Run the tests before you trust anything else:

```
python tests/test_pipeline.py
```

## Step 1 — backtest pipeline

```
pip install -r requirements.txt
python -m slate_lab.ingest --db slate.db --seasons 2023 2024 2025
python -m slate_lab.train  --db slate.db --test-season 2025 --ablation
```

Ingest takes a while (it's polite to a free API — roughly 700 pitcher logs
per season at ~6/second). Training prints three models against each other:

- **baseline** — the exact closed-form model the PWA runs
- **logistic** — linear model over the feature table
- **gbdt** — small, regularized gradient boosting

Validation is walk-forward **by season**: train on everything before the
test season, never after. The `--ablation` flag adds feature groups one at
a time (team strength → recent form → rest → starters) and prints held-out
log loss at each step. A group that doesn't lower the number doesn't earn
a place. This is the feedback loop, done without self-deception.

**How to read results — the numbers that actually matter:**

| | log loss |
|---|---|
| coin flip (always 50%) | 0.6931 |
| always home at 54% | 0.6899 |
| team record only | ~0.6835 |
| **team + starter (current best)** | **0.6827** |
| a PERFECT forecaster | ~0.673–0.680 |

That last row is the important one and it surprises people. Most MLB games
are close to coin flips: a game whose true probability is 55% costs a
perfect predictor 0.688; even a lopsided 65% game costs 0.647. So the
entire achievable range between guessing and omniscience is roughly 0.015,
and a model at 0.6827 has already captured a third to two-thirds of it.

Any result below ~0.67 is a data leak, not a breakthrough. Check the
tripwire before celebrating.

## Step 2 — Statcast

```
pip install pybaseball
python -m slate_lab.statcast --db slate.db --seasons 2024 2025
python -m slate_lab.train --db slate.db --test-season 2025 --ablation
```

Adds a 30-day rolling team xwOBA differential (shifted a day so today's
games never see today's pitches). Importing the module registers the
feature group, so the same ablation now answers the only question that
matters: did Statcast improve held-out log loss or not? First download per
season is slow; pybaseball caches locally.

## Step 3 — nightly job

`.github/workflows/nightly.yml` runs at 7 AM ET on a free GitHub Actions
runner: refreshes the current season, retrains, scores today's slate, and
publishes `predictions.json` to the repo's `gh-pages` branch.

Setup: push this folder to a GitHub repo → Settings → Pages → deploy from
`gh-pages` → run the workflow once manually (first run bootstraps three
seasons of history and is slow; after that the DB is cached).

Then open the PWA's `index.html`, set:

```js
const ML_URL = "https://<user>.github.io/<repo>/predictions.json";
```

and redeploy the app. Each game card shows the ML probability alongside
the built-in model. When they disagree, the feature table can tell you why.

## Step 4 — the monetization gate

This step is deliberately not code. Charging money is gated on evidence,
and the gate is:

1. **Skill** — the trained model beats the team-record-only baseline on
   held-out log loss across at least two separate test seasons.
2. **Calibration** — every bucket's predicted vs actual within noise on
   those seasons.
3. **Live record** — the nightly job runs for 60+ days with predictions
   stamped *before* first pitch, graded after, losses included. Backtests
   convince you; only a live record should convince a customer.
4. **Honest positioning** — unless the model beats the closing line
   (see above: it won't), the product is analytics and education, not an
   edge. Mongoose-style "we show the math, we don't sell picks" is the
   defensible posture; anything promising profit is a lie with a
   subscription fee.
5. **A lawyer** — betting-adjacent products for money touch state gambling
   regulation, advertising rules, and consumer protection. This is a real
   conversation with a real attorney before the first dollar.

One sport until the gate is passed. "All sports" is how this dies of
scope; MLB is the easiest sport to model and it isn't passed yet.

## Layout

```
slate_lab/ingest.py     seasons → SQLite (records facts, no logic)
slate_lab/features.py   point-in-time features; the one place leakage could live
slate_lab/statcast.py   optional xwOBA features via pybaseball
slate_lab/models.py     baseline / logistic / gbdt + metrics
slate_lab/train.py      walk-forward eval + ablation harness
slate_lab/score.py      retrain on everything, score today, emit JSON
tests/test_pipeline.py  synthetic league + the anti-leakage tripwire
```

## The ledger (odds + live grading)

`slate_lab/ledger.py` closes the feedback loop with real sportsbook prices.

**Books:** DraftKings preferred, Hard Rock Bet fallback, per game — both via
The Odds API (the-odds-api.com). DraftKings is licensed in more states and
carries deeper liquidity, so its closing line is the better benchmark.
Neither book has a public API and scraping them violates their terms; the
aggregator is the legitimate route. Free tier is 500 credits/month; the
schedule below uses ~180.

**Setup (one time):** get a free key at the-odds-api.com → repo Settings →
Secrets → Actions → new secret `ODDS_API_KEY`.

**What runs when (all automatic):**
- 11 AM / 6:30 PM / 9:30 PM ET — odds snapshots. The last snapshot before
  each game's first pitch becomes its closing number.
- 7 AM ET — grade yesterday's predictions against final scores and closing
  odds, file today's predictions, publish `report.json`.

**Why files, not database rows:** every snapshot and prediction is committed
to git. Commit timestamps are tamper-evident proof that predictions were
filed before first pitch — the property a live record needs before anyone
(including you) should trust it. `record` refuses to overwrite an existing
day for the same reason.

**Reading report.json:** `model_logloss` vs `market_logloss` on the same
games is the scoreboard. Expect the market to win, but by less than you'd
think — DraftKings' devigged closing line is probably near 0.675, and the
model is at 0.6827. That gap is small in log-loss terms and still fatal in
betting terms, because you are not competing on log loss: you are competing
against a ~4.5% vig. A model slightly behind the market loses money by
definition; a model slightly ahead of it very likely still loses money after
the cut.

This is why the honest product is analytics and education, not picks. Watch
(a) the gap narrowing and (b) `model_won_disagreements` — when model and
market took opposite sides, who was right. Hundreds of disagreements before
that number means anything.


## Results log

**2025 holdout (n=2123), trained on 2023–2024**

| model | log loss | vs coin flip |
|---|---|---|
| hand-tuned formula (the PWA's) | 0.6905 | +0.6% |
| logistic, team + starter | **0.6827** | +2.0% |
| gbdt | 0.6828 | +2.0% |

**Ship logistic.** gbdt tied it, and a tie means the trees are fitting noise.

**Feature findings**

| feature | corr w/ outcome | corr w/ pyth_diff | verdict |
|---|---|---|---|
| pyth_diff | +0.133 | 1.000 | core |
| sp_edge | +0.081 | +0.309 | keep — real but partly redundant |
| l10_diff | +0.070 | — | **cut**, raised log loss |
| rest_diff | −0.014 | — | **cut**, no signal |
| sp_vs_rot | −0.015 | −0.394 | **cut**, moving baseline |

`sp_vs_rot` was an attempt to make the starter feature orthogonal to team
strength by rating a pitcher against his own rotation. It failed because the
baseline moves with team quality — a #3 on a great staff looks bad, a #1 on a
bad staff looks good — so it encodes roster shape, not pitching. Instructive
failure, kept in LEGACY as a warning.

**Next candidates:** park factors, weather, confirmed lineups, pitcher-level
Statcast. Each carries information neither records nor ERA contain. Expect
0.001–0.003 each, and let the ablation reject the ones that don't deliver.
