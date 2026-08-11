# SLATE — Project README

Transparent sports analytics with a public, timestamped, graded record.
The brand is the discipline: predictions are filed before games, graded
against real sportsbook closing lines, and never edited. Losses stay on
the board. Updated 2026-08-10.

---

## The two repos

| Repo | What it is | Where it serves |
|---|---|---|
| **slate-app** | The PWA (single index.html + sw.js) | https://StayOccupiedVR.github.io/slate-app/ |
| **slate-lab** | Models, ledgers, GitHub Actions workflow | data feeds on its gh-pages branch |

Data feeds published by slate-lab, read by the app:
`predictions.json` · `odds.json` (MLB, 3 books) · `nfl-odds.json` ·
`report.json` (graded MLB record)

Deploys are `git push` for both repos. GitHub Pages rebuilds in ~1 min.
The Netlify era is over.

---

## What is live

**MLB (in season)**
- Model: `team + kbb` — Pythagorean team strength + K-BB% starter rating,
  prior-regressed. Holdout log loss **0.6827** on 2025 (n=2123); perfect
  forecaster ≈ .673, coin flip .693. ERA-based starter rating and park
  factors retired to LEGACY with their numbers documented in features.py.
- Ledger: filed daily 7:00 AM ET before first pitch, graded nightly
  against DraftKings closing. Record accumulates in `report.json`.
- Odds: DraftKings + FanDuel + Hard Rock captured 11 AM / 6:30 / 9:30 ET
  (~2 credits per capture, The Odds API free tier, ~500/mo budget).
- App board: sportsbook dark theme, team-color chips, win% pills,
  three-book odds table with best price in green, weather (Open-Meteo,
  30 park table), scratch alerts (regular missing from posted lineup),
  bullpen availability, K-BB% row ("what the model uses"), EV verdict
  prefilled from best available line, ET timestamps for stats + odds.

**NFL (built, sleeping until Week 1 — Sept 9)**
- Adapter validated against embedded closing lines (nflverse carries 27
  seasons of spreads + moneylines): model .60–.65 vs market .56–.62 on
  2023–25 holdouts. Features: Pythagorean (exp 2.37, 6-game prior),
  rest, QB-change flags, divisional.
- EPA session findings (documented in sports/nfl.py, reproducible via
  experiments/nfl_epa.py): team EPA **rejected** (r=0.971 with pyth —
  same signal); QB EPA **benched** (helps 2023/24, hurts 2025 in every
  formulation).
- Weekly ledger (`nfl_ledger.py`): file Wednesdays, revise Sundays
  **only** for QB changes (original preserved beside revision), grade
  Tuesdays with log loss vs devigged close **and CLV** (line movement
  toward our side after filing — accumulates signal every game).
- Eight crons live in the workflow, all behind a season guard that
  checks nflverse for a REG game within 10 days and prints "asleep"
  otherwise. Odds capture is deliberately **unguarded** (display-only,
  never touches the record) — preseason scoreboard shows three-book
  prices via `nfl-odds.json`.
- App: NFL scoreboard route (#nfl) from ESPN's public API — preseason
  display-only with an honest banner; ESPN↔nflverse code aliases
  (WSH→WAS, LAR→LA).

**Props — Phase 1 (research, zero cost)**
- `props.py`: MLB starter strikeout **distributions** (not averages):
  K-per-BF prior-regressed over 70 BF, mixed over full-career empirical
  workloads. Workload-window ablation documented in the constant (last-10
  lost to naive, full career won).
- Real 2025 backtest (n=4,573 starts): **mid-range calibration is good**
  (pred 32→34, 50→48, 30→28, 27→26); high-confidence buckets run ~5pts
  hot (manager-pull ceiling effect — v2 candidate); log score ties naive
  Poisson (2.2369 vs 2.2371). Ships as research display only.
- v2 priorities, evidence-ranked: workload/pitch-count ceiling, opponent
  lineup K%. **No edge claims until prop lines exist** (paid tier gate).

---

## The homepage roadmap (app tiles)

MLB (LIVE) · NFL (PRESEASON→LIVE Sept) · UFC (roadmap) · PGA (roadmap) ·
NHL (roadmap — goalies as the starting-pitcher analog, October) · NCAA
Football (roadmap — NFL adapter + CFBD) · Player props (in development —
cross-sport hub, per-sport subpages, "graded projections not picks").

Capture budget when all live (worst month): MLB 2/day + NFL 6/wk +
NCAAF 4/wk + PGA 2/wk + UFC 2/wk ≈ **328 of 500** free credits.

---

## Gates (non-negotiable)

1. Features ship only past a backtest/ablation; rejects go to LEGACY
   with their numbers.
2. Display and model are separate; the help text says which is which.
3. No edge claims without real lines to claim against.
4. Monetization: 60+ days live record, and **attorney review before any
   paid tier or app-store submission** (odds apps are a regulated
   category on both stores).
5. Prop odds are the first-dollar decision (~$59/mo tier) — triggered by
   record maturity, not calendar.

---

## Delivery workflow (hard-won rules)

- **One PowerShell command per line.** Glued pastes fail.
- **Four beats:** download → `dir` confirm → copy → `Select-String` gate.
  The copy is the beat that gets skipped; False at the gate means redo it.
- **Unique zip names** (`slate-app-vN.zip`, `slate-lab-<change>.zip`) —
  Windows suffixes (`_17`) made same-name zips a stale-file trap.
- **`.github` files ship as named single files** (`nightly-vN.yml`) —
  PowerShell's `*` wildcard skips dot-folders. Burned twice; permanent rule.
- OneDrive sometimes locks `.git` temp dirs mid-rebase: answer `n`, then
  `Remove-Item .git\rebase-merge -Recurse -Force` if needed.
- Push rejections are usually the workflow bot; `git pull --rebase` then
  push.
- Service worker: shell cached, **data feeds never cached** (odds.json,
  nfl-odds.json, predictions.json, report.json, MLB/ESPN/Open-Meteo APIs).
  Bump `slate-vN` in sw.js on every app release; users need two loads.

## Test suites (all must pass before any lab push)

```
python tests/test_pipeline.py     # point-in-time + tampering tripwire
python tests/test_sports.py       # adapter contract + synthetic NFL
python tests/test_ledger.py       # MLB ledger + odds export
python tests/test_nfl_ledger.py   # weekly filing/revision/CLV lifecycle
python tests/test_props.py        # strikeout distribution + calibration
```

## Near-term queue

1. MLB `report.json` review at ~200 graded games
2. Props hub UI (#props/mlb): distribution bars, over-probabilities,
   L10 logs, methodology note
3. NFL spread/margin model (27 years of closing spreads to judge it)
4. Sept: watch Week 1 auto-file; flip NFL tile to LIVE
5. Logo: banner deployed; square original remains for icon redraw
