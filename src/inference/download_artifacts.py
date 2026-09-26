"""Download the deployment checkpoint and scaler from MLflow."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from dotenv import load_dotenv

from src.inference.predict import (
    PROJECT_ROOT,
    default_artifact_paths,
    load_config,
)


LOGGER = logging.getLogger(__name__)
RUN_ID_VARIABLE = "MLFLOW_RUN_ID"
DEPLOYMENT_ARTIFACTS = {
    "checkpoint": "training_outputs/best_model.pth",
    "scaler": "training_outputs/scaler.joblib",
}


def normalize_tracking_uri(value: str) -> str:
    """Return an HTTP(S) tracking URI, matching training configuration."""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1].strip()
    value = value.rstrip("/")
    if not value:
        raise ValueError("The MLflow tracking URI is empty")
    if "://" not in value:
        value = f"http://{value}"
    if not value.startswith(("http://", "https://")):
        raise ValueError("The MLflow tracking URI must use HTTP or HTTPS")
    return value


def resolve_run_id(
    client: Any,
    experiment_name: str,
    requested_run_id: str | None,
) -> str:
    """Resolve an explicit run, or the latest finished Data 4 run."""
    if requested_run_id:
        run = client.get_run(requested_run_id)
        if run.info.status != "FINISHED":
            raise RuntimeError(
                f"MLflow run {requested_run_id} is {run.info.status}, not FINISHED"
            )
        if run.data.tags.get("stage") != "data4":
            raise RuntimeError(
                f"MLflow run {requested_run_id} is not tagged stage=data4"
            )
        return requested_run_id

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment not found: {experiment_name}")
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=(
            "tags.stage = 'data4' and attributes.status = 'FINISHED'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(
            f"No finished stage=data4 run found in {experiment_name}"
        )
    return str(runs[0].info.run_id)


def download_deployment_artifacts() -> str:
    """Download and atomically install the model artifacts required by API."""
    config = load_config()
    load_dotenv(PROJECT_ROOT / ".env")
    mlflow_config = config["mlflow"]
    uri_variable = mlflow_config["tracking_uri_environment_variable"]
    raw_uri = os.getenv(uri_variable)
    if not raw_uri:
        raise RuntimeError(f"{uri_variable} is not configured")
    tracking_uri = normalize_tracking_uri(raw_uri)

    import mlflow
    from mlflow.tracking import MlflowClient

    # Required for runs:/ and mlflow-artifacts:/ artifact resolution.
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    requested_run_id = os.getenv(RUN_ID_VARIABLE, "").strip() or None
    run_id = resolve_run_id(
        client,
        str(mlflow_config["experiment_name"]),
        requested_run_id,
    )

    checkpoint_path, scaler_path = default_artifact_paths(config)
    targets = {
        "checkpoint": checkpoint_path,
        "scaler": scaler_path,
    }
    target_directory = checkpoint_path.parent
    target_directory.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=".mlflow-download-",
        dir=target_directory,
    ) as temporary_directory:
        temporary_path = Path(temporary_directory)
        staged: dict[Path, Path] = {}
        for name, artifact_path in DEPLOYMENT_ARTIFACTS.items():
            LOGGER.info("Downloading %s from MLflow run %s", artifact_path, run_id)
            downloaded = Path(
                client.download_artifacts(
                    run_id,
                    artifact_path,
                    str(temporary_path),
                )
            )
            if not downloaded.is_file() or downloaded.stat().st_size == 0:
                raise RuntimeError(
                    f"MLflow returned an invalid artifact: {artifact_path}"
                )
            staged_path = temporary_path / f"ready-{downloaded.name}"
            shutil.copy2(downloaded, staged_path)
            staged[targets[name]] = staged_path

        for target, staged_path in staged.items():
            staged_path.replace(target)

        run_id_path = temporary_path / "mlflow_run_id.txt"
        run_id_path.write_text(f"{run_id}\n", encoding="utf-8")
        run_id_path.replace(target_directory / "mlflow_run_id.txt")

    LOGGER.info("Installed deployment artifacts from MLflow run %s", run_id)
    return run_id


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    run_id = download_deployment_artifacts()
    print(f"Downloaded MLflow deployment artifacts from run {run_id}")


if __name__ == "__main__":
    main()


__all__ = [
    "DEPLOYMENT_ARTIFACTS",
    "download_deployment_artifacts",
    "normalize_tracking_uri",
    "resolve_run_id",
]
