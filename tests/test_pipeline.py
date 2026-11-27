import json

import pytest

from nbapred import cli, train
from tests.conftest import FIXTURE


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    train.main(["--data", str(FIXTURE), "--train-from", "2012", "--test-season", "2015",
                "--model-path", str(out / "model.joblib"), "--reports", str(out), "--no-plots"])
    return out


def test_train_writes_results_for_all_models(trained):
    res = json.loads((trained / "results.json").read_text())
    assert set(res["test"]) == {"season_avg", "elo", "ridge", "gbm"}
    assert res["val_season"] == 2014 and res["test_season"] == 2015
    for r in res["test"].values():
        assert 5 < r["margin_mae"] < 20 and 0.5 < r["winner_acc"] < 0.85


def test_cli_predicts_plausible_score(trained, capsys):
    h, a = cli.predict(["--home", "BOS", "--away", "LAL", "--date", "2015-03-01",
                     "--data", str(FIXTURE), "--model", str(trained / "model.joblib")])
    assert 80 < h < 130 and 80 < a < 130
    assert "LAL" in capsys.readouterr().out


def test_cli_rejects_unknown_team(trained):
    with pytest.raises(SystemExit):
        cli.main(["--home", "XXX", "--away", "LAL", "--date", "2015-03-01",
                  "--data", str(FIXTURE), "--model", str(trained / "model.joblib")])
