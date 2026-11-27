"""Train, compare against baselines on held-out seasons, and write reports.

    nbapred-train [--source stitched|538|kaggle] [--data GAMES.csv] [--val-season 2024] [--test-season 2025]

Choices (recency half-life, GBM settings, which model the CLI uses) are made
on --val-season with models fit on the seasons before it. Each reported
season (the test season and every later one) is then predicted by models fit
on all seasons before it, so no reported game is seen in training.
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from . import combine, data, ingame, kaggle_data
from .features import build_features
from .models import (ScoreModel, elo_baseline, gbm, metrics, paired_bootstrap, recency_weights, ridge,
                     season_average)

MODEL_PATH = data.ROOT / "models" / "model.joblib"
PROCESSED = data.ROOT / "data" / "processed" / "games.csv"
REPORTS = data.ROOT / "reports"
VAL_SEASON, TEST_SEASON = 2024, 2025
HALF_LIVES = [None, 10, 20]
GBM_GRID = [dict(max_iter=200, max_depth=3), dict(max_iter=400, max_depth=3), dict(max_iter=300, max_depth=5)]
BASELINES = {"season_avg": season_average, "elo": elo_baseline}
LABELS = {"season_avg": "Season average", "elo": "Elo margin", "ridge": "Ridge regression",
          "gbm": "Gradient boosting", "ingame": "In-game ridge", "pace": "Pace extrapolation",
          "pregame_share": "Score + pre-game share"}


def load_table(path: Path) -> pd.DataFrame:
    games = pd.read_csv(path, dtype={"game_id": str, "home": str, "away": str}, parse_dates=["date"])
    return games.sort_values(["date", "game_id"], kind="stable").reset_index(drop=True)


def split(games: pd.DataFrame, val_season: int, test_season: int):
    """Tuning split: train before val_season, validate on it, test on test_season."""
    s = games.season
    return s < val_season, s == val_season, s == test_season


def fit_model(make, games, feats, mask, half_life):
    w = recency_weights(games.season[mask], half_life)
    return ScoreModel(make).fit(feats[mask], games.home_pts[mask], games.away_pts[mask], w)


def score(pred, games, feats, mask) -> dict:
    return metrics(games.home_pts[mask], games.away_pts[mask], *pred(feats[mask]))


def mean_mae(r: dict) -> float:
    return (r["home_mae"] + r["away_mae"]) / 2


def tune(games, feats, val_season):
    tr, va = games.season < val_season, games.season == val_season
    val_score = lambda make, hl: score(fit_model(make, games, feats, tr, hl), games, feats, va)
    by_hl = {hl: val_score(ridge, hl) for hl in HALF_LIVES}
    hl = min(HALF_LIVES, key=lambda h: mean_mae(by_hl[h]))
    by_params = [(p, val_score(gbm(**p), hl)) for p in GBM_GRID]
    params, gbm_val = min(by_params, key=lambda x: mean_mae(x[1]))
    val = {k: score(f, games, feats, va) for k, f in BASELINES.items()} | {"ridge": by_hl[hl], "gbm": gbm_val}
    pick = min(["ridge", "gbm"], key=lambda k: mean_mae(val[k]))
    info = {"val_season": val_season, "train_seasons": [int(games.season[tr].min()), val_season - 1],
            "n_train": int(tr.sum()), "n_val": int(va.sum()), "half_life": hl, "gbm_params": params,
            "cli_model": pick, "val": val,
            "val_half_life_mae": {str(h): round(mean_mae(r), 3) for h, r in by_hl.items()}}
    return info, {"ridge": ridge, "gbm": gbm(**params)}


def gaps(pred, games, feats, te) -> dict:
    """Pick-minus-baseline loss per game, with bootstrap intervals."""
    m_true = (games.home_pts - games.away_pts)[te].to_numpy()
    t_true = (games.home_pts + games.away_pts)[te].to_numpy()

    def losses(f):
        h, a = f(feats[te])
        return {"margin_ae": abs(h - a - m_true), "total_ae": abs(h + a - t_true),
                "winner_right": ((h - a > 0) == (m_true > 0)).astype(float)}

    ours = losses(pred)
    return {b: {k: paired_bootstrap(ours[k], v[k]) for k, v in [(k, losses(f)) for k in ours]}
            for b, f in BASELINES.items()}


def season_eval(games, feats, season, learned, tuning):
    """Fit on every season before `season`, score `season` pre-game and in-game."""
    hl, pick = tuning["half_life"], tuning["cli_model"]
    tr, te = games.season < season, games.season == season
    models = {k: fit_model(m, games, feats, tr, hl) for k, m in learned.items()}
    res = {k: score(f, games, feats, te) for k, f in BASELINES.items()}
    res |= {k: score(m, games, feats, te) for k, m in models.items()}
    entry = {"season": int(season), "n": int(te.sum()), "complete": bool(games.playoffs[te].any()),
             "last_date": f"{games.date[te].max():%Y-%m-%d}", "results": res,
             "cli_model_minus_baseline": gaps(models[pick], games, feats, te), "ingame": {}}

    ingame_models = {}
    for cp in ingame.CHECKPOINTS:
        X_tr, X_te = ingame.frame(games[tr], feats[tr], cp), ingame.frame(games[te], feats[te], cp)
        if X_tr.empty or X_te.empty:
            continue
        y = lambda X: (games.home_pts[X.index], games.away_pts[X.index])
        m = ingame.InGameModel(cp).fit(X_tr, *y(X_tr), recency_weights(games.season[X_tr.index], hl))
        ingame_models[cp] = m
        preds = {"ingame": m, "pace": ingame.pace_baseline(cp),
                 "pregame_share": ingame.pregame_share_baseline(cp, models[pick])}
        entry["ingame"][cp] = {"n": len(X_te)} | {k: metrics(*y(X_te), *f(X_te)) for k, f in preds.items()}
    return entry, models, ingame_models


def by_decade(games, feats, half_life) -> list[dict]:
    """Walk-forward by decade: ridge fit on all earlier seasons, scored on each decade of seasons."""
    rows = []
    first = (int(games.season.min()) // 10 + 1) * 10
    for start in range(first, int(games.season.max()) + 1, 10):
        tr, te = games.season < start, (games.season >= start) & (games.season < start + 10)
        preds = BASELINES | {"ridge": fit_model(ridge, games, feats, tr, half_life)}
        for name, f in preds.items():
            r = score(f, games, feats, te)
            rows.append({"decade": start, "model": name, "n": int(te.sum()),
                         "margin_mae": r["margin_mae"], "total_mae": r["total_mae"]})
    return rows


def results_table(res: dict) -> str:
    lines = ["| Model | Home pts MAE | Away pts MAE | Margin MAE | Total MAE | Winner acc. |",
             "| --- | --- | --- | --- | --- | --- |"]
    for k, r in res.items():
        if k == "n":
            continue
        lines.append(f"| {LABELS[k]} | {r['home_mae']:.2f} | {r['away_mae']:.2f} | "
                     f"{r['margin_mae']:.2f} | {r['total_mae']:.2f} | {r['winner_acc']:.1%} |")
    return "\n".join(lines)


def report_md(info) -> str:
    t = info["tuning"]
    out = [f"## Validation {t['val_season']} (fit on {t['train_seasons'][0]}-{t['train_seasons'][1]})",
           results_table(t["val"])]
    for e in info["seasons"]:
        kind = "season" if e["complete"] else f"season to {e['last_date']} (in progress)"
        out += [f"## {e['season']} {kind}, {e['n']} games, pre-game", results_table(e["results"])]
        for cp, r in e["ingame"].items():
            out += [f"### {e['season']} in-game from {'halftime' if cp == 'half' else 'end of Q3'}, "
                    f"{r['n']} games", results_table(r)]
    if info["decades"]:
        rows = [f"| {r['decade']}-{r['decade'] + 9} | {r['n']} | {LABELS[r['model']]} | "
                f"{r['margin_mae']:.2f} | {r['total_mae']:.2f} |" for r in info["decades"]]
        out += ["## Error by decade (walk-forward)",
                "\n".join(["| Seasons | Games | Model | Margin MAE | Total MAE |", "| --- | --- | --- | --- | --- |"]
                          + rows)]
    return "\n\n".join(out) + "\n"


def load_source(source: str):
    if source == "538":
        return data.load_games(data.download()).assign(source="538"), None
    if source == "kaggle":
        return kaggle_data.load_games(kaggle_data.download()).assign(source="kaggle"), None
    return combine.build()


def main(argv=None):
    p = argparse.ArgumentParser(description="Train and evaluate NBA score models.")
    p.add_argument("--source", choices=["stitched", "538", "kaggle"], default="stitched")
    p.add_argument("--data", type=Path, default=None, help="games table written by an earlier run (skips --source)")
    p.add_argument("--val-season", type=int, default=None, help=f"default {VAL_SEASON}, or test season - 1")
    p.add_argument("--test-season", type=int, default=None, help=f"default {TEST_SEASON} if in the data, else latest")
    p.add_argument("--model-path", type=Path, default=MODEL_PATH)
    p.add_argument("--processed", type=Path, default=PROCESSED, help="where to write the games table")
    p.add_argument("--reports", type=Path, default=REPORTS)
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args(argv)

    overlap = None
    if args.data:
        games = load_table(args.data)
    else:
        games, overlap = load_source(args.source)
        args.processed.parent.mkdir(parents=True, exist_ok=True)
        games.to_csv(args.processed, index=False, float_format="%.4g")
    seasons = set(games.season)
    test_season = args.test_season or (TEST_SEASON if TEST_SEASON in seasons else max(seasons))
    val_season = args.val_season or test_season - 1

    feats = build_features(games)
    tuning, learned = tune(games, feats, val_season)
    entries = []
    for s in sorted(x for x in seasons if x >= test_season):
        entry, models, ingame_models = season_eval(games, feats, s, learned, tuning)
        entries.append(entry)
    info = {"tuning": tuning, "seasons": entries, "decades": by_decade(games, feats, tuning["half_life"]),
            "overlap": overlap}

    # The saved model is the one fit for the newest season: every season before it.
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"name": tuning["cli_model"], "model": models[tuning["cli_model"]], "ingame": ingame_models,
                 "trained_through": int(entries[-1]["season"]) - 1}, args.model_path)
    args.reports.mkdir(parents=True, exist_ok=True)
    (args.reports / "results.json").write_text(json.dumps(info, indent=2, default=str))
    md = report_md(info)
    (args.reports / "results.md").write_text(md)
    if "source" in games:
        combine.coverage(games).to_csv(args.reports / "coverage.csv")
    if not args.no_plots:
        from . import plots
        test = games.season == test_season
        plots.mae_by_model(entries[0], args.reports)
        plots.margin_scatter(fit_model(learned[tuning["cli_model"]], games, feats, games.season < test_season,
                                       tuning["half_life"]), games, feats, test, entries[0],
                             LABELS[tuning["cli_model"]], args.reports)
        if info["decades"]:
            plots.by_decade(info["decades"], args.reports)
    print(md)
    print(f"saved {tuning['cli_model']} model to {args.model_path}")


if __name__ == "__main__":
    main()
