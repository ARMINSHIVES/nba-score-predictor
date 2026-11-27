"""predict --home BOS --away LAL --date 2015-03-01"""
import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

from . import data
from .features import FEATURES, replay
from .train import MODEL_PATH


def season_of(date: pd.Timestamp) -> int:
    """Seasons are named by the year they end; a season starts in October."""
    return date.year + 1 if date.month >= 8 else date.year


def predict(argv=None) -> tuple[float, float]:
    """Parse arguments, print the prediction and return (home_pts, away_pts)."""
    p = argparse.ArgumentParser(prog="predict", description="Predict an NBA final score from pre-game data.")
    p.add_argument("--home", required=True, help="home team code, e.g. BOS")
    p.add_argument("--away", required=True, help="away team code, e.g. LAL")
    p.add_argument("--date", required=True, help="game date, YYYY-MM-DD")
    p.add_argument("--neutral", action="store_true")
    p.add_argument("--playoffs", action="store_true")
    p.add_argument("--data", type=Path, default=data.RAW_PATH)
    p.add_argument("--model", type=Path, default=MODEL_PATH)
    args = p.parse_args(argv)

    if not args.model.exists():
        sys.exit(f"no model at {args.model}; run nbapred-train first")
    date = pd.Timestamp(args.date)
    games = data.load_games(args.data)
    teams = data.team_lookup(games)
    codes = [args.home.upper(), args.away.upper()]
    unknown = [c for c in codes if c not in teams]
    if unknown:
        sys.exit(f"unknown team code(s) {unknown}; known: {', '.join(sorted(teams))}")
    home, away = (teams[c] for c in codes)
    if home == away:
        sys.exit("home and away are the same franchise")

    state = replay(games, until=date)
    row = state.features(home, away, date, season_of(date), int(args.neutral), int(args.playoffs))
    saved = joblib.load(args.model)
    h, a = saved["model"](pd.DataFrame([row], columns=FEATURES))
    h, a = float(h[0]), float(a[0])

    last = games.date.max()
    if date > last:
        print(f"note: data ends {last:%Y-%m-%d}; ratings are as of that date")
    print(f"{codes[1]} {a:.0f} @ {codes[0]} {h:.0f}   "
          f"(margin {h - a:+.1f} home, total {h + a:.1f}, model: {saved['name']})")
    return h, a


def main(argv=None):
    predict(argv)


if __name__ == "__main__":
    main()
