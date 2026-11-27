from pathlib import Path

import pytest

from nbapred.data import load_games

FIXTURE = Path(__file__).parent / "fixtures" / "games_2012_2015.csv"


@pytest.fixture(scope="session")
def games():
    return load_games(FIXTURE)
