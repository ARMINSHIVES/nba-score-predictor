from nbapred.data import team_lookup


def test_one_row_per_game_sorted(games_538):
    assert games_538.game_id.is_unique
    assert games_538.date.is_monotonic_increasing
    assert set(games_538.season) == {2015}


def test_home_away_orientation(games_538):
    # 2015 Finals game 6: Warriors won 105-97 at Cleveland.
    g = games_538.set_index("game_id").loc["201506170CLE"]
    assert (g.home, g.away, g.home_pts, g.away_pts) == ("Cavaliers", "Warriors", 97, 105)
    assert g.playoffs == 1 and g.neutral == 0


def test_team_codes_and_aliases(games_538):
    teams = team_lookup(games_538)
    assert teams["BOS"] == "Celtics"
    assert teams["BKN"] == teams["BRK"] == "Nets"
    assert teams["PHX"] == teams["PHO"] == "Suns"
