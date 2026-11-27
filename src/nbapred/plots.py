"""Report plots. Matplotlib is imported here so training without plots doesn't need it."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID, BLUE, GRAY, ORANGE = "#191919", "#666666", "#e5e5e5", "#257de0", "#b3b3b3", "#e07b25"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})


def _labels():
    from .train import LABELS
    return LABELS


def _tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)


def mae_by_model(entry: dict, out: Path):
    res, labels = entry["results"], _labels()
    names = list(res)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, key, title in zip(axes, ["margin_mae", "total_mae"], ["Margin MAE (points)", "Total MAE (points)"]):
        vals = [res[n][key] for n in names]
        best = min(vals)
        # Dots, not bars: the axis does not start at zero, and bar length would mislead.
        ys = range(len(names))
        ax.hlines(ys, best - 0.4, vals, color=GRID, lw=1)
        ax.scatter(vals, ys, s=70, zorder=3, color=[BLUE if v == best else GRAY for v in vals])
        ax.set_yticks(list(ys), [labels[n] for n in names])
        for i, v in enumerate(vals):
            ax.text(v + 0.08, i, f"{v:.2f}", va="center", color=INK)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlim(best - 0.4, max(vals) + 0.5)
        ax.invert_yaxis()
        ax.xaxis.grid(True, color=GRID)
        _tidy(ax)
    fig.suptitle(f"Held-out {entry['season']} season, {entry['n']} games (lower is better)",
                 x=0.01, ha="left", color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "mae_by_model.png", dpi=150)
    plt.close(fig)


def margin_scatter(model, games, feats, te, entry: dict, label: str, out: Path):
    h, a = model(feats[te])
    pred, true = h - a, (games.home_pts - games.away_pts)[te].to_numpy()
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.scatter(pred, true, s=9, color=BLUE, alpha=0.35, linewidths=0)
    lim = [-45, 45]
    ax.plot(lim, lim, color=MUTED, lw=1, ls="--")
    ax.text(14, 19, "perfect prediction", color=MUTED, fontsize=8, rotation=38)
    ax.axhline(0, color=GRID, lw=1, zorder=0)
    ax.axvline(0, color=GRID, lw=1, zorder=0)
    ax.set_xlim(-25, 25)
    ax.set_ylim(*lim)
    ax.set_xlabel("Predicted home margin")
    ax.set_ylabel("Actual home margin")
    ax.set_title(f"{label}, {entry['season']} season", loc="left")
    _tidy(ax)
    fig.tight_layout()
    fig.savefig(out / "margin_pred_vs_actual.png", dpi=150)
    plt.close(fig)


def by_decade(rows: list[dict], out: Path):
    labels = _labels()
    colors = {"season_avg": GRAY, "elo": ORANGE, "ridge": BLUE}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharex=True)
    for ax, key, title in zip(axes, ["margin_mae", "total_mae"], ["Margin MAE (points)", "Total MAE (points)"]):
        for name, color in colors.items():
            pts = [(r["decade"], r[key]) for r in rows if r["model"] == name]
            ax.plot(*zip(*pts), marker="o", ms=4, lw=1.5, color=color, label=labels[name])
        ax.set_title(title, loc="left", color=INK)
        ax.yaxis.grid(True, color=GRID)
        ax.set_xticks(sorted({r["decade"] for r in rows}),
                      [f"{d}s" for d in sorted({r["decade"] for r in rows})], rotation=45)
        _tidy(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Walk-forward error by decade of seasons: each decade predicted by a model fit on all earlier "
                 "seasons", x=0.01, ha="left", color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "error_by_decade.png", dpi=150)
    plt.close(fig)
