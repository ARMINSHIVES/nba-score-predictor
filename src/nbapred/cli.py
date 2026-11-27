"""predict --home BOS --away LAL --date 2026-12-01 [--halftime 55-60 | --q3 80-84]"""
import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

from . import data
from .features import FEATURES, replay
from .train import MODEL_PATH, PROCESSED, load_table


def season_of(date: pd.Timestamp) -> int:
    """Seasons are named by the year they end; a season starts in October."""
    return date.year + 1 if date.month >= 8 else date.year


def score_arg(text: str) -> tuple[int, int]:
    try:
        h, a = (int(x) for x in text.split("-"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected HOME-AWAY points, e.g. 55-60; got {text!r}")
    return h, a


def predict(argv=None) -> tuple[float, float]:
    """Parse arguments, print the prediction and return (home_pts, away_pts)."""
    p = argparse.ArgumentParser(prog="predict", description="Predict an NBA final score.")
    p.add_argument("--home", required=True, help="home team code, e.g. BOS")
    p.add_argument("--away", required=True, help="away team code, e.g. LAL")
    p.add_argument("--date", required=True, help="game date, YYYY-MM-DD")
    p.add_argument("--neutral", action="store_true")
    p.add_argument("--playoffs", action="store_true")
    now = p.add_mutually_exclusive_group()
    now.add_argument("--halftime", type=score_arg, metavar="H-A", help="score at halftime, home first")
    now.add_argument("--q3", type=score_arg, metavar="H-A", help="score after three quarters, home first")
    p.add_argument("--data", type=Path, default=PROCESSED, help="games table written by nbapred-train")
    p.add_argument("--model", type=Path, default=MODEL_PATH)
    args = p.parse_args(argv)

    for path in (args.model, args.data):
        if not path.exists():
            sys.exit(f"{path} not found; run nbapred-train first")
    date = pd.Timestamp(args.date)
    games = load_table(args.data)
    teams = data.team_lookup(games)
    codes = [args.home.upper(), args.away.upper()]
    unknown = [c for c in codes if c not in teams]
    if unknown:
        sys.exit(f"unknown team code(s) {unknown}; known: {', '.join(sorted(teams))}")
    home, away = (teams[c] for c in codes)
    if home == away:
        sys.exit("home and away are the same franchise")

    state = replay(games, until=date)
    X = pd.DataFrame([state.features(home, away, date, season_of(date), int(args.neutral), int(args.playoffs))],
                     columns=FEATURES)
    saved = joblib.load(args.model)
    cp, score = ("half", args.halftime) if args.halftime else ("q3", args.q3) if args.q3 else (None, None)
    if cp:
        if cp not in saved["ingame"]:
            sys.exit(f"this model has no in-game mode for {cp}")
        h, a = saved["ingame"][cp](X.assign(h_now=score[0], a_now=score[1]))
        how = f"in-game from {'halftime' if cp == 'half' else 'end of Q3'} {score[0]}-{score[1]}"
    else:
        h, a = saved["model"](X)
        how = f"pre-game, model: {saved['name']}"
    h, a = float(h[0]), float(a[0])

    last = games.date.max()
    if date > last + pd.Timedelta(days=1):
        print(f"note: data ends {last:%Y-%m-%d}; ratings are as of that date")
    print(f"{codes[1]} {a:.0f} @ {codes[0]} {h:.0f}   (margin {h - a:+.1f} home, total {h + a:.1f}, {how})")
    return h, a


def main(argv=None):
    predict(argv)


if __name__ == "__main__":
    main()
