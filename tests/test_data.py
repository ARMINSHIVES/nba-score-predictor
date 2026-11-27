from nbapred.data import team_lookup


def test_one_row_per_game_sorted(games):
    assert games.game_id.is_unique
    assert games.date.is_monotonic_increasing
    assert set(games.season) == {2012, 2013, 2014, 2015}


def test_home_away_orientation(games):
    # 2015 Finals game 6: Warriors won 105-97 at Cleveland.
    g = games.set_index("game_id").loc["201506170CLE"]
    assert (g.home, g.away, g.home_pts, g.away_pts) == ("Cavaliers", "Warriors", 97, 105)
    assert g.playoffs == 1 and g.neutral == 0


def test_team_codes_and_aliases(games):
    teams = team_lookup(games)
    assert teams["BOS"] == "Celtics"
    assert teams["BKN"] == teams["BRK"] == teams["NJN"] == "Nets"
    assert teams["PHX"] == "Suns"
