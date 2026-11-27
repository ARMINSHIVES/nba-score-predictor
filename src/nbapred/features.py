"""Pre-game features built by replaying games in date order.

All games on a date get their features from the state at the end of the
previous date; only then are that date's results applied. So no feature can
see a same-day or later result. tests/test_leakage.py checks this.
"""
from collections import deque
from dataclasses import dataclass, field

import pandas as pd

ELO_INIT = 1500.0
ELO_MEAN = 1505.0
ELO_K = 20.0
ELO_HCA = 100.0
ELO_CARRY = 0.75   # share of last season's rating kept at a new season
SHRINK = 8.0       # games of prior weight on season averages
RECENT = 10
REST_CAP = 10

FEATURES = [
    "elo_diff", "neutral", "playoffs", "league_avg",
    "h_elo", "h_for", "h_against", "h_l10_for", "h_l10_against", "h_rest", "h_games",
    "a_elo", "a_for", "a_against", "a_l10_for", "a_l10_against", "a_rest", "a_games",
]


@dataclass
class Team:
    elo: float = ELO_INIT
    season: int = 0
    last_date: pd.Timestamp | None = None
    n: int = 0
    pts_for: float = 0.0
    pts_against: float = 0.0
    prior_for: float | None = None
    prior_against: float | None = None
    recent: deque = field(default_factory=lambda: deque(maxlen=RECENT))


def mov_mult(margin: float, elo_diff_winner: float) -> float:
    return (abs(margin) + 3) ** 0.8 / (7.5 + 0.006 * elo_diff_winner)


def elo_shift(home_elo, away_elo, margin, neutral):
    """Rating points the home team gains (away team loses the same)."""
    hca = 0.0 if neutral else ELO_HCA
    diff = home_elo + hca - away_elo
    expected = 1 / (1 + 10 ** (-diff / 400))
    won = 1.0 if margin > 0 else 0.0
    winner_diff = diff if margin > 0 else -diff
    return ELO_K * mov_mult(margin, winner_diff) * (won - expected)


class State:
    def __init__(self):
        self.teams: dict[str, Team] = {}
        self.league: dict[int, list[float]] = {}   # season -> [points, team-games]

    def league_avg(self, season: int) -> float:
        cur = self.league.get(season, [0.0, 0])
        prev = self.league.get(season - 1)
        prior = prev[0] / prev[1] if prev and prev[1] else 100.0
        return (cur[0] + SHRINK * prior) / (cur[1] + SHRINK)

    def team(self, name: str, season: int) -> Team:
        t = self.teams.setdefault(name, Team(season=season))
        if t.season != season:  # season rollover, known before tip-off
            if t.n:
                t.prior_for, t.prior_against = t.pts_for / t.n, t.pts_against / t.n
            t.elo = ELO_CARRY * t.elo + (1 - ELO_CARRY) * ELO_MEAN
            t.season, t.n, t.pts_for, t.pts_against = season, 0, 0.0, 0.0
        return t

    def _side(self, t: Team, date, league: float) -> list[float]:
        pf = t.prior_for if t.prior_for is not None else league
        pa = t.prior_against if t.prior_against is not None else league
        s_for = (t.pts_for + SHRINK * pf) / (t.n + SHRINK)
        s_against = (t.pts_against + SHRINK * pa) / (t.n + SHRINK)
        if t.recent:
            k = len(t.recent)
            l_for = sum(r[0] for r in t.recent) / k
            l_against = sum(r[1] for r in t.recent) / k
        else:
            l_for, l_against = s_for, s_against
        rest = REST_CAP if t.last_date is None else min((date - t.last_date).days, REST_CAP)
        return [t.elo, s_for, s_against, l_for, l_against, rest, t.n]

    def features(self, home, away, date, season, neutral=0, playoffs=0) -> list[float]:
        league = self.league_avg(season)
        h = self._side(self.team(home, season), date, league)
        a = self._side(self.team(away, season), date, league)
        return [h[0] - a[0], neutral, playoffs, league] + h + a

    def update(self, g) -> None:
        h, a = self.team(g.home, g.season), self.team(g.away, g.season)
        shift = elo_shift(h.elo, a.elo, g.home_pts - g.away_pts, g.neutral)
        h.elo += shift
        a.elo -= shift
        for t, pf, pa in ((h, g.home_pts, g.away_pts), (a, g.away_pts, g.home_pts)):
            t.n += 1
            t.pts_for += pf
            t.pts_against += pa
            t.recent.append((pf, pa))
            t.last_date = g.date
        lg = self.league.setdefault(g.season, [0.0, 0])
        lg[0] += g.home_pts + g.away_pts
        lg[1] += 2


def replay(games: pd.DataFrame, until=None) -> State:
    """State after every game strictly before `until` (all games if None)."""
    state = State()
    if until is not None:
        games = games[games.date < pd.Timestamp(until)]
    for g in games.itertuples(index=False):
        state.update(g)
    return state


def build_features(games: pd.DataFrame) -> pd.DataFrame:
    state = State()
    ordered = games.sort_values("date", kind="stable")
    rows, pending = [], []
    for g in ordered.itertuples(index=False):
        # Apply a date's results only once the next date starts.
        if pending and pending[0].date != g.date:
            for p in pending:
                state.update(p)
            pending = []
        rows.append(state.features(g.home, g.away, g.date, g.season, g.neutral, g.playoffs))
        pending.append(g)
    return pd.DataFrame(rows, columns=FEATURES, index=ordered.index).loc[games.index]
