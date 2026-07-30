# slate-lab — one-command entry points
DB=slate.db
TEST_SEASON=2025

test:            ## prove the pipeline is leak-free before anything else
	python3 tests/test_pipeline.py

ingest:          ## pull three seasons (slow once; polite to the free API)
	python3 -m slate_lab.ingest --db $(DB) --seasons 2023 2024 2025

backtest:        ## walk-forward eval + feature ablation
	python3 -m slate_lab.train --db $(DB) --test-season $(TEST_SEASON) --ablation

statcast:        ## optional: add xwOBA features, then re-run backtest
	python3 -m slate_lab.statcast --db $(DB) --seasons 2024 2025
	python3 -m slate_lab.train --db $(DB) --test-season $(TEST_SEASON) --ablation

score:           ## score today's slate locally -> predictions.json
	python3 -m slate_lab.score --db $(DB) --update-season

all: test ingest backtest
