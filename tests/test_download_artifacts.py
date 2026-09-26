from types import SimpleNamespace

import pytest

from src.inference.download_artifacts import (
    normalize_tracking_uri,
    resolve_run_id,
)


def run(run_id="data4-run", status="FINISHED", stage="data4"):
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id, status=status),
        data=SimpleNamespace(tags={"stage": stage}),
    )


class FakeClient:
    def __init__(self, runs=None):
        self.runs = runs or []

    def get_run(self, run_id):
        return self.runs[0]

    def get_experiment_by_name(self, name):
        return SimpleNamespace(experiment_id="experiment-1")

    def search_runs(self, *args, **kwargs):
        return self.runs


def test_tracking_uri_adds_http_scheme():
    assert normalize_tracking_uri("mlflow.internal:5000/") == (
        "http://mlflow.internal:5000"
    )
    assert normalize_tracking_uri("'https://mlflow.internal:5000/'") == (
        "https://mlflow.internal:5000"
    )


def test_explicit_run_must_be_finished_data4_run():
    assert resolve_run_id(
        FakeClient([run()]), "soh-training", "data4-run"
    ) == "data4-run"

    with pytest.raises(RuntimeError, match="not tagged stage=data4"):
        resolve_run_id(
            FakeClient([run(stage="lab")]),
            "soh-training",
            "lab-run",
        )


def test_latest_finished_data4_run_is_selected():
    assert resolve_run_id(
        FakeClient([run(run_id="latest")]),
        "soh-training",
        None,
    ) == "latest"
