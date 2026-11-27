"""Stitch FiveThirtyEight (1946-47 to 2014-15) and Kaggle (1946-47 to today) into one game table.

Kaggle is missing most games before 1954-55, so those seasons come from
FiveThirtyEight; every later season comes from Kaggle, which also has box
scores and quarter scores. Both sources cover 1954-55 to 2014-15, and
`overlap_report` checks them against each other game by game.
"""
import pandas as pd

from . import data, kaggle_data

LAST_538_SEASON = 1954

# FiveThirtyEight franchise -> NBA team id, for franchises that play on past 1953-54.
# Franchises that folded keep a "538:" key.
FRAN_TO_ID = {
    "Celtics": "1610612738", "Hawks": "1610612737", "Kings": "1610612758", "Knicks": "1610612752",
    "Lakers": "1610612747", "Pistons": "1610612765", "Sixers": "1610612755", "Warriors": "1610612744",
}


def fran_key(fran: pd.Series) -> pd.Series:
    return fran.map(FRAN_TO_ID).fillna("538:" + fran)


def stitch(old: pd.DataFrame, new: pd.DataFrame, last_old: int = LAST_538_SEASON) -> pd.DataFrame:
    old = old[old.season <= last_old].assign(home=lambda d: fran_key(d.home), away=lambda d: fran_key(d.away),
                                             source="538")
    new = new[new.season > last_old].assign(source="kaggle")
    games = pd.concat([old, new], ignore_index=True)
    return games.sort_values(["date", "game_id"], kind="stable").reset_index(drop=True)


def _team_sides(g: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game: (game, season, date, team, points scored, points allowed)."""
    cols = ["game_id", "season", "date"]
    return pd.concat([
        g[cols + ["home", "home_pts", "away_pts"]].set_axis(cols + ["team", "pf", "pa"], axis=1),
        g[cols + ["away", "away_pts", "home_pts"]].set_axis(cols + ["team", "pf", "pa"], axis=1)])


def season_team_map(old: pd.DataFrame, new: pd.DataFrame) -> pd.Series:
    """(538 franchise, season) -> NBA team id, by majority vote over games with the same date and score.

    Done per season because the two sources split some histories differently
    (FiveThirtyEight files the 1988-2002 Charlotte Hornets under the Pelicans).
    """
    o, n = _team_sides(old), _team_sides(new)
    m = o.merge(n, on=["date", "pf", "pa"], suffixes=("", "_k"))
    return m.groupby(["team", "season"]).team_k.agg(lambda s: s.mode().iloc[0])


def overlap_report(old: pd.DataFrame, new: pd.DataFrame) -> dict:
    """Compare the sources on the seasons both cover.

    A game matches when the same two teams meet within a day of each other
    (one source sometimes dates a game a day off). Then the final scores and
    the home/away order are compared.
    """
    seasons = sorted(set(old.season) & set(new.season))
    o, n = old[old.season.isin(seasons)].copy(), new[new.season.isin(seasons)].copy()
    tmap = season_team_map(o, n)
    for side in ("home", "away"):
        o[side] = pd.MultiIndex.from_arrays([o[side], o.season]).map(tmap)
    o = o.dropna(subset=["home", "away"])

    def keyed(g):
        pair = g[["home", "away"]].apply(sorted, axis=1, result_type="expand")
        return g.assign(t1=pair[0], t2=pair[1])

    o, n = keyed(o), keyed(n)
    n_used = set()
    matched = []
    n_by_key = n.groupby(["t1", "t2", "date"]).groups
    for g in o.itertuples(index=False):
        for shift in (0, -1, 1):
            hits = [i for i in n_by_key.get((g.t1, g.t2, g.date + pd.Timedelta(days=shift)), []) if i not in n_used]
            if hits:
                i = hits[0]
                n_used.add(i)
                k = n.loc[i]
                same = {g.home: g.home_pts, g.away: g.away_pts} == {k.home: k.home_pts, k.away: k.away_pts}
                matched.append((g.game_id, k.game_id, shift, g.home == k.home, same,
                                f"{g.home_pts}-{g.away_pts}", f"{k.home_pts}-{k.away_pts}"))
                break
    m = pd.DataFrame(matched, columns=["id_538", "id_kaggle", "day_shift", "same_home", "same_score",
                                     "score_538", "score_kaggle"])
    return {
        "seasons": [seasons[0], seasons[-1]] if seasons else [],
        "games_538": len(old[old.season.isin(seasons)]), "games_kaggle": len(n),
        "matched": len(m), "same_score": int(m.same_score.sum()),
        "score_differs": int((~m.same_score).sum()), "home_away_swapped": int((~m.same_home).sum()),
        "date_off_by_one": int((m.day_shift != 0).sum()),
        "unmatched_538": len(old[old.season.isin(seasons)]) - len(m), "unmatched_kaggle": len(n) - len(m),
        "score_differs_examples": m[~m.same_score].head(5).to_dict("records"),
    }


def coverage(games: pd.DataFrame) -> pd.DataFrame:
    """Per-season game counts and how many games carry quarter scores and box-score possessions."""
    g = games.groupby("season")
    return pd.DataFrame({
        "source": g.source.agg(lambda s: "/".join(sorted(set(s)))),
        "games": g.size(), "regular": g.playoffs.agg(lambda p: int((p == 0).sum())),
        "playoffs": g.playoffs.sum(), "with_halftime": g.h_half.count(), "with_possessions": g.h_poss.count(),
    })


def build() -> tuple[pd.DataFrame, dict]:
    """Download both sources if needed, stitch them and check their overlap."""
    old = data.load_games(data.download())
    new = kaggle_data.load_games(kaggle_data.download())
    report = overlap_report(old[old.season > LAST_538_SEASON], new[new.season <= old.season.max()])
    return stitch(old, new), report
