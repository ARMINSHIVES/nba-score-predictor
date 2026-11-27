from pathlib import Path

import pytest

from nbapred import data, kaggle_data
from nbapred.train import load_table

FIXTURES = Path(__file__).parent / "fixtures"
GAMES = FIXTURES / "games_2024_2026.csv"   # normalized table, as nbapred-train writes it
RAW_538 = FIXTURES / "games_538_2015.csv"  # FiveThirtyEight rows, 2014-15 season
RAW_KAGGLE = FIXTURES / "kaggle"           # a few hand-picked Kaggle games, raw format


@pytest.fixture(scope="session")
def games():
    return load_table(GAMES)


@pytest.fixture(scope="session")
def games_538():
    return data.load_games(RAW_538)


@pytest.fixture(scope="session")
def games_kaggle():
    return kaggle_data.load_games(RAW_KAGGLE)
