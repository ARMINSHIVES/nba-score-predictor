import pandas as pd
import pytest

from nbapred.features import REST_CAP, State, elo_shift


def test_elo_home_win_gains_and_upsets_gain_more():
    assert elo_shift(1500, 1500, 10, neutral=0) > 0
    assert elo_shift(1500, 1500, -10, neutral=0) < 0
    favourite = elo_shift(1700, 1400, 5, neutral=0)
    underdog = elo_shift(1400, 1700, 5, neutral=0)
    assert underdog > favourite > 0


def test_elo_is_zero_sum():
    s = State()
    g = pd.Series(dict(home="A", away="B", season=2015, date=pd.Timestamp("2014-11-01"),
                       home_pts=100, away_pts=90, neutral=0))
    s.update(g)
    assert s.teams["A"].elo + s.teams["B"].elo == pytest.approx(3000)


def test_first_game_uses_priors():
    s = State()
    f = dict(zip(["elo_diff", "neutral", "playoffs", "league_avg", "h_elo", "h_for", "h_against",
                  "h_l10_for", "h_l10_against", "h_rest", "h_games"],
                 s.features("A", "B", pd.Timestamp("2014-11-01"), 2015)))
    assert f["elo_diff"] == 0
    assert f["h_games"] == 0 and f["h_rest"] == REST_CAP
    assert f["h_for"] == f["league_avg"] == 100.0


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
