"""Download and load eoinamoore's "Historical NBA Data and Player Box Scores" (Kaggle, CC0-1.0).

One row per game in the shared game schema (see GAME_COLUMNS), with halftime
and end-of-Q3 scores and box-score possession estimates where the source has
them. Teams are keyed by NBA team id, which stays the same when a franchise
moves or is renamed (SEA -> OKC, NJN -> BKN).
"""
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ROOT

DATASET = "eoinamoore/historical-nba-data-and-player-box-scores"
RAW_DIR = ROOT / "data" / "raw" / "kaggle"
FILES = ["Games.csv", "TeamStatistics.csv", "TeamHistories.csv"]

GAME_COLUMNS = ["game_id", "date", "season", "playoffs", "neutral", "home", "away",
                "home_code", "away_code", "home_pts", "away_pts",
                "h_half", "a_half", "h_q3", "a_q3", "h_poss", "a_poss", "minutes"]
# gameId digit 3: 1 preseason, 2 regular season, 3 All-Star, 4 playoffs, 5 play-in, 6 NBA Cup final.
DROP_TYPES = {"1", "3"}
PLAYOFF_TYPES = {"4", "5"}
# Box scores cover every game from 1985-86; before that only a few percent of games
# have them, too few to rate teams, and a model fit on them extrapolates wildly.
BOX_FROM = 1986

GAME_COLS = ["gameId", "gameDateTimeEst", "gameSubtype", "gameSubLabel", "arenaCity"]
STAT_COLS = ["gameId", "teamId", "home", "teamScore", "fieldGoalsAttempted", "freeThrowsAttempted",
             "reboundsOffensive", "turnovers", "numMinutes",
             "q1Points", "q2Points", "q3Points", "q4Points", "otAllPoints"]


def download(dest: Path = RAW_DIR, force: bool = False) -> Path:
    """Fetch the needed files with the official Kaggle API (needs the `data` extra and a Kaggle token)."""
    missing = [f for f in FILES if force or not (dest / f).exists()]
    if not missing:
        return dest
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    dest.mkdir(parents=True, exist_ok=True)
    for name in missing:
        api.dataset_download_file(DATASET, name, path=str(dest), force=True, quiet=True)
        zipped = dest / f"{name}.zip"
        if zipped.exists():  # large files arrive zipped
            with zipfile.ZipFile(zipped) as z:
                z.extract(name, dest)
            zipped.unlink()
    return dest


def season_of_id(game_id: pd.Series) -> pd.Series:
    """Season label (year it ends) from gameId digits 4-5, the season's start year.

    Dates can't do this: the 2019-20 season finished in the 2020 bubble in October.
    """
    yy = game_id.str[3:5].astype(int)
    return np.where(yy >= 46, 1900 + yy, 2000 + yy) + 1


def team_codes(histories: pd.DataFrame, team_id: pd.Series, season: pd.Series) -> np.ndarray:
    """The team's abbreviation in force at the start of that season."""
    spans = (histories.assign(code=histories.teamAbbrev.str.strip())
             .sort_values("seasonFounded")[["teamId", "seasonFounded", "code"]])
    q = pd.DataFrame({"teamId": team_id.to_numpy(), "start": (season - 1).to_numpy(),
                      "i": np.arange(len(team_id))}).sort_values("start")
    m = pd.merge_asof(q, spans, left_on="start", right_on="seasonFounded", by="teamId")
    first = spans.drop_duplicates("teamId").set_index("teamId").code  # team older than its first span
    m["code"] = m.code.fillna(m.teamId.map(first))
    return m.sort_values("i").code.to_numpy()


def _quarters_ok(s: pd.DataFrame) -> pd.Series:
    q = s[["q1Points", "q2Points", "q3Points", "q4Points"]]
    total = q.sum(axis=1) + s.otAllPoints.fillna(0)
    return q.notna().all(axis=1) & (total == s.teamScore) & (q.sum(axis=1) > 0)


def _possessions(s: pd.DataFrame) -> pd.Series:
    """Box-score estimate FGA - OREB + TOV + 0.44 FTA; NaN where the box score is missing."""
    ok = (s.fieldGoalsAttempted > 0) & (s.reboundsOffensive > 0) & (s.turnovers > 0)
    poss = s.fieldGoalsAttempted - s.reboundsOffensive + s.turnovers + 0.44 * s.freeThrowsAttempted
    return poss.where(ok)


def load_games(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    games = pd.read_csv(raw_dir / "Games.csv", dtype={"gameId": str}, low_memory=False,
                        usecols=GAME_COLS + ["hometeamId", "awayteamId", "homeScore", "awayScore"])
    games["gameId"] = games.gameId.str.zfill(10)
    kind = games.gameId.str[2]
    games = games[~kind.isin(DROP_TYPES) & (games.homeScore > 0) & (games.awayScore > 0)].copy()

    stats = pd.read_csv(raw_dir / "TeamStatistics.csv", dtype={"gameId": str}, usecols=STAT_COLS)
    stats["gameId"] = stats.gameId.str.zfill(10)
    stats["half"] = (stats.q1Points + stats.q2Points).where(_quarters_ok(stats))
    stats["q3"] = (stats.half + stats.q3Points)
    stats["poss"] = _possessions(stats)
    side = lambda flag: (stats[stats.home == flag]
                         .drop_duplicates("gameId").set_index("gameId")[["half", "q3", "poss", "numMinutes"]])
    h, a = side(1), side(0)
    gid = games.gameId

    date = pd.to_datetime(games.gameDateTimeEst.str[:10])
    season = season_of_id(gid)
    kind = gid.str[2]
    bubble = (season == 2020) & (date >= "2020-07-01")
    neutral = ((kind == "6") | games.gameSubtype.eq("Global Games") | games.arenaCity.eq("Las Vegas")
               | games.gameSubLabel.fillna("").str.contains("Semifinal") | bubble)

    hist = pd.read_csv(raw_dir / "TeamHistories.csv")
    s = pd.Series(season, index=games.index)
    out = pd.DataFrame({
        "game_id": gid.to_numpy(),
        "date": date.to_numpy(),
        "season": season,
        "playoffs": kind.isin(PLAYOFF_TYPES).astype(int).to_numpy(),
        "neutral": neutral.astype(int).to_numpy(),
        "home": games.hometeamId.astype(str).to_numpy(),
        "away": games.awayteamId.astype(str).to_numpy(),
        "home_code": team_codes(hist, games.hometeamId, s),
        "away_code": team_codes(hist, games.awayteamId, s),
        "home_pts": games.homeScore.astype(int).to_numpy(),
        "away_pts": games.awayScore.astype(int).to_numpy(),
        "h_half": h.half.reindex(gid).to_numpy(), "a_half": a.half.reindex(gid).to_numpy(),
        "h_q3": h.q3.reindex(gid).to_numpy(), "a_q3": a.q3.reindex(gid).to_numpy(),
        "h_poss": h.poss.reindex(gid).to_numpy(), "a_poss": a.poss.reindex(gid).to_numpy(),
        "minutes": h.numMinutes.reindex(gid).to_numpy(),
    })
    # A half or Q3 score is usable only when both teams' quarters check out.
    for x in ("half", "q3"):
        bad = out[f"h_{x}"].isna() | out[f"a_{x}"].isna()
        out.loc[bad, [f"h_{x}", f"a_{x}"]] = np.nan
    out.loc[out.season < BOX_FROM, ["h_poss", "a_poss"]] = np.nan
    return out.sort_values(["date", "game_id"], kind="stable").reset_index(drop=True)
