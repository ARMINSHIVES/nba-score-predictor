"""A feature for a game on date D may use only games played before D."""
import numpy as np
import pandas as pd
import pytest

from nbapred import ingame
from nbapred.features import FEATURES, build_features, replay
from nbapred.train import split

# Every per-game result column a feature could learn from. Quarter scores are
# not pre-game features at all, but perturbing them guards against that changing.
RESULTS = {"home_pts": (60, 160), "away_pts": (60, 160), "h_half": (30, 80), "a_half": (30, 80),
           "h_q3": (50, 120), "a_q3": (50, 120), "h_poss": (80, 120), "a_poss": (80, 120), "minutes": (240, 290)}


def perturb(games, cutoff):
    rng = np.random.default_rng(0)
    changed = games.copy()
    later = changed.date >= cutoff
    for col, (lo, hi) in RESULTS.items():
        changed.loc[later, col] = rng.integers(lo, hi, later.sum()).astype(float)
    return changed


@pytest.mark.parametrize("cutoff", ["2024-01-15", "2025-02-10", "2026-04-20"])
def test_features_ignore_same_day_and_later_results(games, cutoff):
    cutoff = pd.Timestamp(cutoff)
    assert (games.date == cutoff).any(), "cutoff must be a game day"
    before, after = build_features(games), build_features(perturb(games, cutoff))

    known = games.date <= cutoff  # includes the cutoff day itself
    pd.testing.assert_frame_equal(before[known], after[known])
    assert not before[~known].equals(after[~known]), "perturbation had no effect; test is vacuous"
    for col in ["h_ortg", "h_drtg", "h_pace", "league_pace", "h_l10_for"]:
        assert not before.loc[~known, col].equals(after.loc[~known, col]), f"{col} never changed"


def test_ingame_inputs_use_only_the_checkpoint_score(games):
    """Changing a game's final score must not change its in-game inputs."""
    cutoff = pd.Timestamp("2025-02-10")
    day = games.date == cutoff
    changed = games.copy()
    changed.loc[day, ["home_pts", "away_pts"]] += 17
    for cp in ingame.CHECKPOINTS:
        a = ingame.frame(games[day], build_features(games)[day], cp)
        b = ingame.frame(changed[day], build_features(changed)[day], cp)
        assert len(a) and list(a.columns) == ingame.COLUMNS
        pd.testing.assert_frame_equal(a, b)


def test_prediction_path_matches_training_features(games):
    feats = build_features(games)
    for i in [100, 2500, len(games) - 1]:
        g = games.iloc[i]
        row = replay(games, until=g.date).features(g.home, g.away, g.date, g.season, g.neutral, g.playoffs)
        assert row == pytest.approx(feats.iloc[i][FEATURES].tolist(), nan_ok=True)


def test_split_is_by_season_and_ordered(games):
    tr, va, te = split(games, 2025, 2026)
    assert not (tr & va).any() and not (va & te).any() and not (tr & te).any()
    assert games.date[tr].max() < games.date[va].min()
    assert games.date[va].max() < games.date[te].min()
