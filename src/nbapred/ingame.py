"""In-game mode: predict the final score from the score at halftime or after three quarters.

This is a different task from the pre-game model and is reported separately.
Inputs are the pre-game features (state before the game day) plus the score
at the checkpoint; the model predicts the points still to come.
"""
import numpy as np
import pandas as pd

from .features import FEATURES
from .models import ScoreModel, ridge

CHECKPOINTS = {"half": 24, "q3": 36}   # minutes played at the checkpoint
NOW = ["h_now", "a_now"]
COLUMNS = FEATURES + NOW


def frame(games: pd.DataFrame, feats: pd.DataFrame, cp: str) -> pd.DataFrame:
    """Pre-game features plus the checkpoint score, for games whose quarter scores check out."""
    ok = games[f"h_{cp}"].notna()
    return feats[ok].assign(h_now=games.loc[ok, f"h_{cp}"], a_now=games.loc[ok, f"a_{cp}"])


class InGameModel:
    def __init__(self, cp: str):
        self.cp, self.rest = cp, ScoreModel(ridge, COLUMNS)

    def fit(self, X, y_home, y_away, weight=None):
        self.rest.fit(X, y_home - X.h_now, y_away - X.a_now, weight)
        return self

    def __call__(self, X):
        h, a = self.rest(X)
        return X.h_now.to_numpy() + h, X.a_now.to_numpy() + a


def pace_baseline(cp: str):
    """Keep scoring at the same rate for the rest of regulation."""
    scale = 48 / CHECKPOINTS[cp]
    return lambda X: (X.h_now.to_numpy() * scale, X.a_now.to_numpy() * scale)


def pregame_share_baseline(cp: str, pregame):
    """Score so far plus the pre-game prediction for the minutes left."""
    left = 1 - CHECKPOINTS[cp] / 48

    def predict(X):
        h, a = pregame(X)
        return X.h_now.to_numpy() + left * np.asarray(h), X.a_now.to_numpy() + left * np.asarray(a)
    return predict
