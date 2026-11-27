"""Download and load the FiveThirtyEight NBA game file (CC BY 4.0)."""
import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "nbaallelo.csv"
URL = "https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-elo/nbaallelo.csv"
SHA256 = "d46ed3540ee8d9eca31b3e94cc8c777e0be5156173d814ebf65b8195e8d616bc"

# Pairs of codes for the same team (NBA.com style, Basketball-Reference style, short forms).
ALIASES = [("BKN", "BRK"), ("PHX", "PHO"), ("CHA", "CHO"), ("NOP", "NOR"), ("NOP", "NOH"),
           ("GSW", "GS"), ("SAS", "SA"), ("SAS", "SAN"), ("NYK", "NY"), ("UTA", "UTAH"), ("WAS", "WSH")]
# Columns the FiveThirtyEight file lacks; NaN keeps the game schema shared with kaggle_data.
EXTRA = ["h_half", "a_half", "h_q3", "a_q3", "h_poss", "a_poss", "minutes"]

COLUMNS = ["game_id", "lg_id", "_iscopy", "year_id", "date_game", "is_playoffs",
           "team_id", "fran_id", "pts", "opp_id", "opp_fran", "opp_pts", "game_location"]


def download(path: Path = RAW_PATH) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(URL, timeout=60) as resp, open(tmp, "wb") as f:
        f.write(resp.read())
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    if digest != SHA256:
        tmp.unlink()
        raise ValueError(f"checksum mismatch for {URL}: got {digest}")
    tmp.rename(path)
    return path


def load_games(path: Path = RAW_PATH) -> pd.DataFrame:
    """One row per NBA game, oriented home/away, sorted by date.

    Teams are keyed by franchise (fran_id) so ratings carry across moves and
    renames (SEA -> OKC, NJN -> BRK). Neutral-site games keep the listed team
    as "home" with neutral=1.
    """
    raw = pd.read_csv(path, usecols=COLUMNS)
    raw = raw[(raw.lg_id == "NBA") & (raw._iscopy == 0)]
    away = raw.game_location == "A"
    pick = lambda a, b: a.where(~away, b).to_numpy()
    games = pd.DataFrame({
        "game_id": raw.game_id.to_numpy(),
        "date": pd.to_datetime(raw.date_game, format="%m/%d/%Y").to_numpy(),
        "season": raw.year_id.to_numpy(),
        "playoffs": raw.is_playoffs.to_numpy(),
        "neutral": (raw.game_location == "N").astype(int).to_numpy(),
        "home": pick(raw.fran_id, raw.opp_fran),
        "away": pick(raw.opp_fran, raw.fran_id),
        "home_code": pick(raw.team_id, raw.opp_id),
        "away_code": pick(raw.opp_id, raw.team_id),
        "home_pts": pick(raw.pts, raw.opp_pts),
        "away_pts": pick(raw.opp_pts, raw.pts),
    })
    games[EXTRA] = float("nan")
    return games.sort_values(["date", "game_id"], kind="stable").reset_index(drop=True)


def team_lookup(games: pd.DataFrame) -> dict:
    """Map team code -> franchise, using each code's latest use (plus aliases)."""
    codes = pd.concat([games[["date", "home_code", "home"]].set_axis(["date", "code", "fran"], axis=1),
                       games[["date", "away_code", "away"]].set_axis(["date", "code", "fran"], axis=1)])
    lookup = codes.sort_values("date").groupby("code").fran.last().to_dict()
    for a, b in ALIASES:
        if a in lookup or b in lookup:
            team = lookup.get(a, lookup.get(b))
            lookup.setdefault(a, team)
            lookup.setdefault(b, team)
    return lookup
