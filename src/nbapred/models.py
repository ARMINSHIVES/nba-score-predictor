"""Baselines, learned models and metrics. Every predictor returns (home_pts, away_pts)."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import ELO_HCA, FEATURES

ELO_PER_POINT = 28.0  # ~28 Elo points per point of expected margin (FiveThirtyEight's rule)


def season_average(X: pd.DataFrame):
    """Each team's season-to-date scoring averaged with the opponent's points allowed."""
    return ((X.h_for + X.a_against) / 2).to_numpy(), ((X.a_for + X.h_against) / 2).to_numpy()


def elo_baseline(X: pd.DataFrame):
    """Margin from the Elo gap, total from the league's season-to-date average."""
    margin = (X.elo_diff + ELO_HCA * (1 - X.neutral)) / ELO_PER_POINT
    total = 2 * X.league_avg
    return ((total + margin) / 2).to_numpy(), ((total - margin) / 2).to_numpy()


class ScoreModel:
    """Two regressors, one per team score; margin and total are derived from them."""

    def __init__(self, make):
        self.home, self.away = make(), make()

    def fit(self, X, y_home, y_away):
        self.home.fit(X[FEATURES], y_home)
        self.away.fit(X[FEATURES], y_away)
        return self

    def __call__(self, X):
        return self.home.predict(X[FEATURES]), self.away.predict(X[FEATURES])


def ridge():
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def gbm(max_iter=300, learning_rate=0.05, max_depth=4):
    return lambda: HistGradientBoostingRegressor(
        max_iter=max_iter, learning_rate=learning_rate, max_depth=max_depth,
        min_samples_leaf=50, l2_regularization=1.0, random_state=0)


def metrics(home_true, away_true, home_pred, away_pred) -> dict:
    home_true, away_true = np.asarray(home_true, float), np.asarray(away_true, float)
    m_true, m_pred = home_true - away_true, home_pred - away_pred
    t_true, t_pred = home_true + away_true, home_pred + away_pred
    return {
        "home_mae": float(np.mean(np.abs(home_pred - home_true))),
        "away_mae": float(np.mean(np.abs(away_pred - away_true))),
        "margin_mae": float(np.mean(np.abs(m_pred - m_true))),
        "total_mae": float(np.mean(np.abs(t_pred - t_true))),
        "winner_acc": float(np.mean((m_pred > 0) == (m_true > 0))),
    }


def paired_bootstrap(loss_a, loss_b, n=5000, seed=0) -> dict:
    """Mean of loss_a - loss_b per game, with a 95% bootstrap interval over games."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    idx = np.random.default_rng(seed).integers(0, len(d), (n, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
    return {"diff": float(d.mean()), "ci95": [float(lo), float(hi)]}
