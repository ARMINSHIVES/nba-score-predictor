import numpy as np
import pandas as pd
import pytest

from nbapred.models import elo_baseline, metrics, season_average


def test_metrics_hand_computed():
    m = metrics([100, 90], [90, 100], np.array([98.0, 95.0]), np.array([95.0, 96.0]))
    # game 1: errors 2 / 5, margin 10 vs 3, total 190 vs 193, winner right
    # game 2: errors 5 / 4, margin -10 vs -1, total 190 vs 191, winner right
    assert m["home_mae"] == pytest.approx(3.5)
    assert m["away_mae"] == pytest.approx(4.5)
    assert m["margin_mae"] == pytest.approx(8.0)
    assert m["total_mae"] == pytest.approx(2.0)
    assert m["winner_acc"] == 1.0


def test_elo_baseline_home_court_and_neutral():
    X = pd.DataFrame({"elo_diff": [0.0, 0.0], "neutral": [0, 1], "league_avg": [100.0, 100.0]})
    h, a = elo_baseline(X)
    assert h[0] - a[0] == pytest.approx(100 / 28)
    assert h[1] == a[1] == 100.0


def test_season_average_mixes_offence_and_defence():
    X = pd.DataFrame({"h_for": [110.0], "h_against": [100.0], "a_for": [90.0], "a_against": [104.0]})
    h, a = season_average(X)
    assert (h[0], a[0]) == (107.0, 95.0)
