import json

import pytest

from nbapred import cli, train
from tests.conftest import GAMES


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    train.main(["--data", str(GAMES), "--val-season", "2025", "--test-season", "2026",
                "--model-path", str(out / "model.joblib"), "--processed", str(out / "games.csv"),
                "--reports", str(out), "--no-plots"])
    return out


def run_cli(trained, *args):
    return cli.predict(["--home", "BOS", "--away", "LAL", "--date", "2026-03-01", *args,
                        "--data", str(GAMES), "--model", str(trained / "model.joblib")])


def test_train_writes_results_for_all_models(trained):
    res = json.loads((trained / "results.json").read_text())
    t = res["tuning"]
    assert t["val_season"] == 2025 and t["train_seasons"] == [2024, 2024]
    assert t["cli_model"] in {"ridge", "gbm"} and t["half_life"] in train.HALF_LIVES
    (season,) = res["seasons"]
    assert season["season"] == 2026 and season["complete"]
    assert set(season["results"]) == {"season_avg", "elo", "ridge", "gbm"}
    for r in season["results"].values():
        assert 5 < r["margin_mae"] < 20 and 0.5 < r["winner_acc"] < 0.85
    assert set(season["ingame"]) == {"half", "q3"}
    for cp in season["ingame"].values():
        assert set(cp) == {"n", "ingame", "pace", "pregame_share"}
        assert cp["ingame"]["margin_mae"] < season["results"]["ridge"]["margin_mae"]


def test_cli_predicts_plausible_score(trained, capsys):
    h, a = run_cli(trained)
    assert 80 < h < 140 and 80 < a < 140
    assert "LAL" in capsys.readouterr().out


def test_cli_halftime_starts_from_the_score(trained, capsys):
    h, a = run_cli(trained, "--halftime", "70-40")
    assert h > 100 and a > 60 and h - a > 15
    assert "halftime 70-40" in capsys.readouterr().out


def test_cli_rejects_unknown_team(trained):
    with pytest.raises(SystemExit):
        cli.main(["--home", "XXX", "--away", "LAL", "--date", "2026-03-01",
                  "--data", str(GAMES), "--model", str(trained / "model.joblib")])
