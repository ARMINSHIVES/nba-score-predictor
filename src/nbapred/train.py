"""Train, compare against baselines on a held-out season, and write reports.

    nbapred-train [--data PATH] [--train-from 1990] [--test-season 2015]
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from . import data
from .features import build_features
from .models import ScoreModel, elo_baseline, gbm, metrics, paired_bootstrap, ridge, season_average

MODEL_PATH = data.ROOT / "models" / "model.joblib"
REPORTS = data.ROOT / "reports"
GBM_GRID = [dict(max_iter=200, max_depth=3), dict(max_iter=400, max_depth=3), dict(max_iter=300, max_depth=5)]
LABELS = {"season_avg": "Season average", "elo": "Elo margin", "ridge": "Ridge regression", "gbm": "Gradient boosting"}


def split(games: pd.DataFrame, train_from: int, test_season: int):
    """Train on [train_from, test-2], validate on test-1, test on test_season."""
    s = games.season
    return (s >= train_from) & (s <= test_season - 2), s == test_season - 1, s == test_season


def evaluate(games, feats, train_from, test_season):
    tr, va, te = split(games, train_from, test_season)
    y = games[["home_pts", "away_pts"]]
    score = lambda pred, m: metrics(y.home_pts[m], y.away_pts[m], *pred(feats[m]))

    def val_mae(model):
        r = score(model, va)
        return (r["home_mae"] + r["away_mae"]) / 2

    fit = lambda make, m: ScoreModel(make).fit(feats[m], y.home_pts[m], y.away_pts[m])
    best_params = min(GBM_GRID, key=lambda p: val_mae(fit(gbm(**p), tr)))

    learned = {"ridge": ridge, "gbm": gbm(**best_params)}
    val = {"season_avg": score(season_average, va), "elo": score(elo_baseline, va)}
    val |= {k: score(fit(make, tr), va) for k, make in learned.items()}

    trva = tr | va
    final = {k: fit(make, trva) for k, make in learned.items()}
    test = {"season_avg": score(season_average, te), "elo": score(elo_baseline, te)}
    test |= {k: score(m, te) for k, m in final.items()}

    pick = min(learned, key=lambda k: (val[k]["home_mae"] + val[k]["away_mae"]) / 2)
    m_true = (y.home_pts - y.away_pts)[te].to_numpy()
    t_true = (y.home_pts + y.away_pts)[te].to_numpy()

    def losses(pred):
        h, a = pred(feats[te])
        return {"margin_ae": abs(h - a - m_true), "total_ae": abs(h + a - t_true),
                "winner_right": ((h - a > 0) == (m_true > 0)).astype(float)}

    ours = losses(final[pick])
    vs = {b: losses(f) for b, f in [("season_avg", season_average), ("elo", elo_baseline)]}
    gaps = {b: {k: paired_bootstrap(ours[k], l[k]) for k in ours} for b, l in vs.items()}
    info = {
        "train_seasons": [train_from, test_season - 2], "val_season": test_season - 1,
        "test_season": test_season, "n_train": int(tr.sum()), "n_val": int(va.sum()),
        "n_test": int(te.sum()), "gbm_params": best_params, "cli_model": pick,
        "val": val, "test": test,
        "cli_model_minus_baseline": gaps,
    }
    return info, final, te


def results_table(res: dict) -> str:
    lines = ["| Model | Home pts MAE | Away pts MAE | Margin MAE | Total MAE | Winner acc. |",
             "| --- | --- | --- | --- | --- | --- |"]
    for k, r in res.items():
        lines.append(f"| {LABELS[k]} | {r['home_mae']:.2f} | {r['away_mae']:.2f} | "
                     f"{r['margin_mae']:.2f} | {r['total_mae']:.2f} | {r['winner_acc']:.1%} |")
    return "\n".join(lines)


def plots(info, model, games, feats, te, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, blue, gray = "#191919", "#666666", "#e5e5e5", "#257de0", "#b3b3b3"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": muted, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted, "text.color": ink})

    names = list(info["test"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, key, title in zip(axes, ["margin_mae", "total_mae"], ["Margin MAE (points)", "Total MAE (points)"]):
        vals = [info["test"][n][key] for n in names]
        best = min(vals)
        # Dots, not bars: the axis does not start at zero, and bar length would mislead.
        ys = range(len(names))
        ax.hlines(ys, min(vals) - 0.4, vals, color=grid, lw=1)
        ax.scatter(vals, ys, s=70, zorder=3, color=[blue if v == best else gray for v in vals])
        ax.set_yticks(list(ys), [LABELS[n] for n in names])
        for i, v in enumerate(vals):
            ax.text(v + 0.08, i, f"{v:.2f}", va="center", color=ink)
        ax.set_title(title, loc="left", color=ink)
        ax.set_xlim(min(vals) - 0.4, max(vals) + 0.4)
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.grid(True, color=grid)
        ax.set_axisbelow(True)
    fig.suptitle(f"Held-out {info['test_season']} season, {info['n_test']} games (lower is better)",
                 x=0.01, ha="left", color=muted, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "mae_by_model.png", dpi=150)
    plt.close(fig)

    h, a = model(feats[te])
    pred, true = h - a, (games.home_pts - games.away_pts)[te].to_numpy()
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.scatter(pred, true, s=9, color=blue, alpha=0.35, linewidths=0)
    lim = [-45, 45]
    ax.plot(lim, lim, color=muted, lw=1, ls="--")
    ax.text(14, 19, "perfect prediction", color=muted, fontsize=8, rotation=38)
    ax.axhline(0, color=grid, lw=1, zorder=0)
    ax.axvline(0, color=grid, lw=1, zorder=0)
    ax.set_xlim(-25, 25)
    ax.set_ylim(*lim)
    ax.set_xlabel("Predicted home margin")
    ax.set_ylabel("Actual home margin")
    ax.set_title(f"{LABELS[info['cli_model']]}, {info['test_season']} season", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "margin_pred_vs_actual.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description="Train and evaluate NBA score models.")
    p.add_argument("--data", type=Path, default=None, help="games CSV (default: download FiveThirtyEight file)")
    p.add_argument("--train-from", type=int, default=1990)
    p.add_argument("--test-season", type=int, default=None, help="default: latest season in the data")
    p.add_argument("--model-path", type=Path, default=MODEL_PATH)
    p.add_argument("--reports", type=Path, default=REPORTS)
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args(argv)

    games = data.load_games(args.data or data.download())
    test_season = args.test_season or int(games.season.max())
    feats = build_features(games)
    info, final, te = evaluate(games, feats, args.train_from, test_season)

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"name": info["cli_model"], "model": final[info["cli_model"]]}, args.model_path)
    args.reports.mkdir(parents=True, exist_ok=True)
    (args.reports / "results.json").write_text(json.dumps(info, indent=2))
    table = results_table(info["test"])
    (args.reports / "results.md").write_text(table + "\n")
    if not args.no_plots:
        plots(info, final[info["cli_model"]], games, feats, te, args.reports)
    print(f"test season {test_season}: {info['n_test']} games")
    print(table)
    print(f"saved {info['cli_model']} model to {args.model_path}")


if __name__ == "__main__":
    main()
