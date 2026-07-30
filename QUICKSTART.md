# Quickstart

## Path A — run locally (fastest way to see results)
    unzip slate-lab.zip && cd slate-lab
    pip install -r requirements.txt
    make test        # must print ALL TESTS PASSED
    make ingest      # ~1-2 hours, one time
    make backtest    # the answer: results.json + console table

## Path B — let GitHub do everything (enables the nightly job)
    cd slate-lab
    git init && git add -A && git commit -m "slate-lab"
    gh repo create slate-lab --private --source . --push
    # (or create the repo on github.com and: git remote add origin <url> && git push -u origin main)

Then on github.com: Actions tab -> nightly-score -> Run workflow.
First run bootstraps 3 seasons (slow); nightly runs after are minutes.
Settings -> Pages -> deploy from branch: gh-pages.
Your predictions URL: https://<user>.github.io/slate-lab/predictions.json

## Wire the app
In slate-app/index.html set:
    const ML_URL = "https://<user>.github.io/slate-lab/predictions.json";
Re-drag the folder onto Netlify Drop. Done — game cards now show the ML number.

## Then
Send me results.json (or paste the console table). The ablation rows decide
what gets built next.
