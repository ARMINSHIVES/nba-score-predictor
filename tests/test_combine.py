import pandas as pd

from nbapred import combine

LAKERS, CELTICS = "1610612747", "1610612738"


def game(gid, date, season, home, away, hp, ap):
    return dict(game_id=gid, date=pd.Timestamp(date), season=season, home=home, away=away,
                home_pts=hp, away_pts=ap, playoffs=0, neutral=0)


def test_overlap_flags_changed_score_swap_and_date_shift():
    # 538 keys teams by franchise name; Kaggle by NBA team id.
    old = pd.DataFrame([
        game("a1", "1960-01-01", 1960, "Lakers", "Celtics", 100, 90),
        game("a2", "1960-01-03", 1960, "Celtics", "Lakers", 110, 105),
        game("a3", "1960-01-05", 1960, "Lakers", "Celtics", 99, 98),   # Kaggle says 99-97
        game("a4", "1960-01-07", 1960, "Lakers", "Celtics", 120, 101),  # Kaggle has it a day later
        game("a5", "1960-01-09", 1960, "Lakers", "Celtics", 95, 93),    # Kaggle has home/away swapped
    ])
    new = pd.DataFrame([
        game("k1", "1960-01-01", 1960, LAKERS, CELTICS, 100, 90),
        game("k2", "1960-01-03", 1960, CELTICS, LAKERS, 110, 105),
        game("k3", "1960-01-05", 1960, LAKERS, CELTICS, 99, 97),
        game("k4", "1960-01-08", 1960, LAKERS, CELTICS, 120, 101),
        game("k5", "1960-01-09", 1960, CELTICS, LAKERS, 93, 95),
        game("k6", "1960-01-11", 1960, LAKERS, CELTICS, 88, 80),       # only in Kaggle
    ])
    r = combine.overlap_report(old, new)
    assert (r["matched"], r["same_score"], r["score_differs"]) == (5, 4, 1)
    assert (r["date_off_by_one"], r["home_away_swapped"]) == (1, 1)
    assert (r["unmatched_538"], r["unmatched_kaggle"]) == (0, 1)
    assert r["score_differs_examples"][0]["id_538"] == "a3"


def test_stitch_takes_early_seasons_from_538_and_maps_franchises():
    old = pd.DataFrame([game("a1", "1950-01-01", 1950, "Lakers", "Capitols", 80, 70),
                        game("a2", "1960-01-01", 1960, "Lakers", "Celtics", 100, 90)])
    new = pd.DataFrame([game("k1", "1950-01-01", 1950, LAKERS, CELTICS, 1, 1),
                        game("k2", "1960-01-01", 1960, LAKERS, CELTICS, 100, 90)])
    s = combine.stitch(old, new, last_old=1954)
    assert list(s.game_id) == ["a1", "k2"] and list(s.source) == ["538", "kaggle"]
    assert (s.home[0], s.away[0]) == (LAKERS, "538:Capitols")


def test_fran_key_keeps_folded_franchises_apart():
    keys = combine.fran_key(pd.Series(["Lakers", "Capitols", "Celtics"]))
    assert list(keys) == [LAKERS, "538:Capitols", CELTICS]
