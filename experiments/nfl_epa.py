"""NFL EPA experiment — 2026-08-04. Reproduces the benched QB-EPA feature.

Run from repo root (network required; pulls nflverse):
    python experiments/nfl_epa.py

Downloads games.csv + play-by-play 2014-2025, aggregates EPA to
team-weeks and qb-weeks, builds point-in-time features, and prints
holdout log loss (base vs +qb_epa vs closing market) for 2023-2025.
Findings and the ship/bench decision are documented in
slate_lab/sports/nfl.py. If a future run with more seasons shows the
2025 damage was season-specific noise, re-open the feature.
"""
import io
import urllib.request
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

GAMES = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
PBP = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{s}.parquet"
COLS = ["season", "week", "posteam", "defteam", "epa", "qb_dropback",
        "passer_player_id", "passer_player_name", "season_type"]


def fetch(url):
    return urllib.request.urlopen(url, timeout=180).read()


def main():
    games = pd.read_csv(io.BytesIO(fetch(GAMES)))
    qbw_frames = []
    for season in range(2014, 2026):
        df = pd.read_parquet(io.BytesIO(fetch(PBP.format(s=season))), columns=COLS)
        df = df[(df.season_type == "REG") & df.epa.notna()]
        qb = df[df.qb_dropback == 1].dropna(subset=["passer_player_id"])
        qbw_frames.append(qb.groupby(
            ["season", "week", "posteam", "passer_player_id",
             "passer_player_name"]).agg(
            db=("epa", "size"), qb_epa=("epa", "sum")).reset_index()
            .rename(columns={"posteam": "team"}))
        print(f"{season}: {len(df)} plays")
    qbw = pd.concat(qbw_frames)

    g = games[(games.season >= 2015) & (games.game_type == "REG")
              & games.away_score.notna()].sort_values(
        ["season", "week"]).reset_index(drop=True)
    name_to_id = {}
    for r in qbw.sort_values(["season", "week"]).itertuples():
        name_to_id[r.passer_player_name] = r.passer_player_id

    def key(n):
        p = n.split() if isinstance(n, str) else []
        return (p[0][0] + "." + p[-1]) if len(p) >= 2 else None

    rows = []
    for season, sg in g.groupby("season", sort=True):
        pf = defaultdict(float); pa = defaultdict(float)
        gp = defaultdict(int); qb_last = {}
        qb_tot = defaultdict(lambda: [0.0, 0.0])
        for r in qbw[qbw.season < season].itertuples():
            w = 0.5 ** (season - 1 - r.season)
            qb_tot[r.passer_player_id][0] += r.qb_epa * w
            qb_tot[r.passer_player_id][1] += r.db * w
        qbn = qbw[qbw.season == season]

        def qr(nm):
            pid = name_to_id.get(key(nm))
            if pid is None:
                return 0.0
            s_, n_ = qb_tot.get(pid, [0.0, 0.0])
            return s_ / (n_ + 200) if n_ > 0 else 0.0

        def py(t):
            f, a = pf[t] + 126, pa[t] + 126
            return f ** 2.37 / (f ** 2.37 + a ** 2.37)

        for r in sg.itertuples():
            if gp[r.away_team] >= 3 and gp[r.home_team] >= 3:
                mla, mlh = r.away_moneyline, r.home_moneyline
                mkt = None
                if pd.notna(mla) and pd.notna(mlh):
                    p = lambda m: (-m) / ((-m) + 100) if m < 0 else 100 / (m + 100)
                    mkt = p(mla) / (p(mla) + p(mlh))
                rows.append(dict(
                    season=season,
                    pyth_diff=py(r.away_team) - py(r.home_team),
                    qb_epa_diff=qr(r.away_qb_name) - qr(r.home_qb_name),
                    rest_diff=(r.away_rest - r.home_rest)
                    if pd.notna(r.away_rest) else 0,
                    qb_new_away=int(qb_last.get(r.away_team) is not None
                                    and qb_last.get(r.away_team) != r.away_qb_name),
                    qb_new_home=int(qb_last.get(r.home_team) is not None
                                    and qb_last.get(r.home_team) != r.home_qb_name),
                    div=int(r.div_game) if pd.notna(r.div_game) else 0,
                    away_won=int(r.away_score > r.home_score),
                    market_p_away=mkt))
            pf[r.away_team] += r.away_score; pa[r.away_team] += r.home_score
            gp[r.away_team] += 1
            pf[r.home_team] += r.home_score; pa[r.home_team] += r.away_score
            gp[r.home_team] += 1
            qb_last[r.away_team] = r.away_qb_name
            qb_last[r.home_team] = r.home_qb_name
            for q in qbn[(qbn.week == r.week)
                         & qbn.team.isin((r.away_team, r.home_team))].itertuples():
                qb_tot[q.passer_player_id][0] += q.qb_epa
                qb_tot[q.passer_player_id][1] += q.db

    F = pd.DataFrame(rows).dropna(subset=["market_p_away"])
    base = ["pyth_diff", "rest_diff", "qb_new_away", "qb_new_home", "div"]
    print("\nseason   base     +qb_epa   market")
    for season in (2023, 2024, 2025):
        tr, te = F[F.season < season], F[F.season == season]
        b = log_loss(te.away_won, LogisticRegression(C=1.0).fit(
            tr[base], tr.away_won).predict_proba(te[base])[:, 1])
        cols = base + ["qb_epa_diff"]
        q = log_loss(te.away_won, LogisticRegression(C=1.0).fit(
            tr[cols], tr.away_won).predict_proba(te[cols])[:, 1])
        mkt = log_loss(te.away_won, te.market_p_away)
        print(f"{season}    {b:.4f}   {q:.4f}   {mkt:.4f}")


if __name__ == "__main__":
    main()
