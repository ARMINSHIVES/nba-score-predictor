"""A feature for a game on date D may use only games played before D."""
import numpy as np
import pandas as pd
import pytest

from nbapred.features import FEATURES, build_features, replay
from nbapred.train import split


@pytest.mark.parametrize("cutoff", ["2013-01-15", "2014-04-16", "2015-06-04"])
def test_features_ignore_same_day_and_later_results(games, cutoff):
    cutoff = pd.Timestamp(cutoff)
    assert (games.date == cutoff).any(), "cutoff must be a game day"
    before = build_features(games)

    rng = np.random.default_rng(0)
    changed = games.copy()
    later = changed.date >= cutoff
    changed.loc[later, "home_pts"] = rng.integers(60, 160, later.sum())
    changed.loc[later, "away_pts"] = rng.integers(60, 160, later.sum())
    after = build_features(changed)

    known = games.date <= cutoff  # includes the cutoff day itself
    pd.testing.assert_frame_equal(before[known], after[known])
    assert not before[~known].equals(after[~known]), "perturbation had no effect; test is vacuous"


def test_prediction_path_matches_training_features(games):
    feats = build_features(games)
    for i in [100, 2500, len(games) - 1]:
        g = games.iloc[i]
        row = replay(games, until=g.date).features(g.home, g.away, g.date, g.season, g.neutral, g.playoffs)
        assert row == pytest.approx(feats.iloc[i][FEATURES].tolist())


def test_split_is_by_season_and_ordered(games):
    tr, va, te = split(games, 2012, 2015)
    assert not (tr & va).any() and not (va & te).any() and not (tr & te).any()
    assert games.date[tr].max() < games.date[va].min()
    assert games.date[va].max() < games.date[te].min()
