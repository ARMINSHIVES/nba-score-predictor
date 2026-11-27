import math
from types import SimpleNamespace

import pandas as pd
import pytest

from nbapred.features import FEATURES, NAN, REST_CAP, State, elo_shift, game_rates


def test_elo_home_win_gains_and_upsets_gain_more():
    assert elo_shift(1500, 1500, 10, neutral=0) > 0
    assert elo_shift(1500, 1500, -10, neutral=0) < 0
    favourite = elo_shift(1700, 1400, 5, neutral=0)
    underdog = elo_shift(1400, 1700, 5, neutral=0)
    assert underdog > favourite > 0


def test_elo_is_zero_sum():
    s = State()
    s.update(game())
    assert s.teams["A"].elo + s.teams["B"].elo == pytest.approx(3000)


def test_first_game_uses_priors():
    s = State()
    f = dict(zip(FEATURES, s.features("A", "B", pd.Timestamp("2014-11-01"), 2015)))
    assert f["elo_diff"] == 0
    assert f["h_games"] == 0 and f["h_rest"] == REST_CAP
    assert f["h_for"] == f["league_avg"] == 100.0
    assert math.isnan(f["h_ortg"]) and math.isnan(f["league_pace"])   # no box scores seen yet


def game(**kw):
    base = dict(home="A", away="B", season=2024, date=pd.Timestamp("2023-11-01"), neutral=0, playoffs=0,
                home_pts=110, away_pts=100, h_poss=100.0, a_poss=100.0, minutes=240.0)
    return SimpleNamespace(**(base | kw))


def test_game_rates_per_100_possessions_and_overtime_pace():
    assert game_rates(game()) == pytest.approx((110.0, 100.0, 100.0))
    # Overtime: 265 team-minutes, so pace is scaled back to 48 minutes.
    assert game_rates(game(minutes=265.0))[2] == pytest.approx(100 * 240 / 265)
    assert game_rates(game(h_poss=NAN)) is None and game_rates(game(minutes=NAN)) is None


def test_box_rates_accumulate_and_skip_games_without_box_scores():
    s = State()
    s.update(game())
    s.update(game(date=pd.Timestamp("2023-11-03"), h_poss=NAN, a_poss=NAN, home_pts=150))
    f = dict(zip(FEATURES, s.features("A", "B", pd.Timestamp("2023-11-05"), 2024)))
    assert s.teams["A"].n == 2 and s.teams["A"].n_box == 1
    # League ORtg so far is (110 + 100) / 2 = 105; one box-score game is shrunk toward it by SHRINK=8 games.
    assert f["h_ortg"] == pytest.approx((110 + 8 * 105) / 9) and f["a_ortg"] == pytest.approx((100 + 8 * 105) / 9)
    assert f["h_drtg"] == pytest.approx((100 + 8 * 105) / 9)
    assert f["h_pace"] == pytest.approx(100.0) and f["league_pace"] == pytest.approx(100.0)


def test_season_totals_match_raw_scores(games):
    first = games.iloc[:200]
    s = State()
    for g in first.itertuples(index=False):
        s.update(g)
    name = first.home.iloc[0]
    scored = pd.concat([first[first.home == name].home_pts, first[first.away == name].away_pts])
    t = s.teams[name]
    assert t.n == len(scored)
    assert t.pts_for / t.n == pytest.approx(scored.mean())
