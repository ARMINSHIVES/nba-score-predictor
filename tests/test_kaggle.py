import numpy as np

from nbapred import kaggle_data
from nbapred.data import team_lookup

CLE, GSW, OKC = "1610612739", "1610612744", "1610612760"


def test_drops_preseason_and_all_star(games_kaggle):
    kinds = set(games_kaggle.game_id.str[2])
    assert kinds == {"2", "4", "5", "6"}
    assert games_kaggle.game_id.is_unique and games_kaggle.date.is_monotonic_increasing


def test_finals_game_orientation_and_season(games_kaggle):
    # 2015 Finals game 6, 16 June 2015: Warriors won 105-97 at Cleveland.
    g = games_kaggle.set_index("game_id").loc["0041400406"]
    assert (g.home, g.away, g.home_pts, g.away_pts) == (CLE, GSW, 97, 105)
    assert (g.season, g.playoffs, g.neutral) == (2015, 1, 0)
    assert (g.home_code, g.away_code) == ("CLE", "GSW")


def test_quarter_scores_are_consistent(games_kaggle):
    q = games_kaggle.dropna(subset=["h_half"])
    assert len(q) > 10
    for side, pts in (("h", "home_pts"), ("a", "away_pts")):
        assert (q[f"{side}_half"] <= q[f"{side}_q3"]).all() and (q[f"{side}_q3"] <= q[pts]).all()


def test_missing_quarters_and_old_box_scores_are_nan(games_kaggle):
    no_quarters = games_kaggle[games_kaggle.season == 2022]
    assert len(no_quarters) and no_quarters[["h_half", "a_q3"]].isna().all().all()
    old = games_kaggle[games_kaggle.season < kaggle_data.BOX_FROM]
    assert len(old) and old[["h_poss", "a_poss"]].isna().all().all()
    new = games_kaggle[games_kaggle.season >= 2023]
    assert (new.h_poss.between(80, 130)).all()


def test_neutral_sites(games_kaggle):
    g = games_kaggle.set_index("game_id")
    assert g.loc[["0041900406", "0021901315"]].neutral.eq(1).all()   # 2020 bubble
    assert g.loc["0062500001"].neutral == 1                          # NBA Cup final, Las Vegas
    assert g.loc["0022500602"].neutral == 1                          # Global Games
    assert g.loc["0052400211"].playoffs == 1 and g.loc["0052400211"].neutral == 0


def test_relocated_team_keeps_its_key(games_kaggle):
    g = games_kaggle.set_index("game_id")
    sea, okc = g.loc["0020701230"], g.loc["0022501196"]
    code = lambda r: r.home_code if r.home == OKC else r.away_code
    assert OKC in (sea.home, sea.away) and OKC in (okc.home, okc.away)
    assert (code(sea), code(okc)) == ("SEA", "OKC")
    teams = team_lookup(games_kaggle)
    assert teams["SEA"] == teams["OKC"] == OKC


def test_season_of_id_handles_century():
    import pandas as pd
    ids = pd.Series(["0024600001", "0029900001", "0020000001", "0042500001"])
    assert list(kaggle_data.season_of_id(ids)) == [1947, 2000, 2001, 2026]
    assert np.issubdtype(kaggle_data.season_of_id(ids).dtype, np.integer)
