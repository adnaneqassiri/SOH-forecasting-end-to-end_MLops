"""Config-driven lab pretraining and Data 4 fine-tuning with MLflow."""

from contextlib import nullcontext
import json
import logging
import os
from pathlib import Path
import random
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from src.data.datasets import (
    DATA4_FEATURES_COLS,
    LAB_FEATURES_COLS,
    PREDICTION_LENGTH,
    create_data4_multimodal_datasets,
    create_lab_multimodal_datasets,
    load_data4_features,
    load_lab_features,
    soh_to_image,
)
from src.model.model import Multimodal, load_pretrained_multimodal
from src.training.evaluate import evaluate_model, load_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
ENV_PATH = PROJECT_ROOT / ".env"
LOGGER = logging.getLogger(__name__)


def load_config():
    with CONFIG_PATH.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {CONFIG_PATH}")
    return config


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was configured but is not available")
    return torch.device(name)


def split_group_ids(dataframe, split_config, seed):
    ids = dataframe["vehicle_id"].drop_duplicates().to_numpy(dtype=int)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    train_fraction = float(split_config["train_fraction"])
    validation_fraction = float(split_config["validation_fraction"])
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("Split fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must sum to less than one")
    train_end = max(1, int(len(ids) * train_fraction))
    validation_end = train_end + max(1, int(len(ids) * validation_fraction))
    if validation_end >= len(ids):
        raise ValueError("At least three groups are required for dataset splitting")
    return {
        "train": ids[:train_end].tolist(),
        "val": ids[train_end:validation_end].tolist(),
        "test": ids[validation_end:].tolist(),
    }


def build_datasets(stage, config, seed):
    split_config = config["training"]["splits"][stage]
    if stage == "lab":
        dataframe = load_lab_features()
        split = split_group_ids(dataframe, split_config, seed)
        datasets = create_lab_multimodal_datasets(
            train_ids=split["train"],
            val_id=split["val"],
            test_id=split["test"],
            soh_to_image=soh_to_image,
            feature_cols=LAB_FEATURES_COLS,
            return_scaler=True,
        )
        return datasets, LAB_FEATURES_COLS, split

    dataframe = load_data4_features()
    split = split_group_ids(dataframe, split_config, seed)
    datasets = create_data4_multimodal_datasets(
        train_ids=split["train"],
        val_id=split["val"],
        test_id=split["test"],
        soh_to_image=soh_to_image,
        feature_cols=DATA4_FEATURES_COLS,
        return_scaler=True,
    )
    return datasets, DATA4_FEATURES_COLS, split


def create_loaders(datasets, stage_config, training_config, device, seed):
    workers = int(training_config["num_workers"])
    common = {
        "batch_size": int(stage_config["batch_size"]),
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = datasets
    return (
        DataLoader(train_dataset, shuffle=True, generator=generator, **common),
        DataLoader(val_dataset, shuffle=False, **common),
        DataLoader(test_dataset, shuffle=False, **common),
    )


def mean_image_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for input_image, sequence, target_image, _, _, _ in loader:
            input_image = input_image.to(device, non_blocking=True)
            sequence = sequence.to(device, non_blocking=True)
            target_image = target_image.to(device, non_blocking=True)
            loss = criterion(model(input_image, sequence), target_image)
            batch_size = input_image.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    return total_loss / total_samples


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    stage_config,
    training_config,
    checkpoint_path,
    device,
    mlflow_module,
):
    criterion = nn.MSELoss()
    best_loss = float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, int(stage_config["epochs"]) + 1):
        started = time.time()
        model.train()
        for module_name in getattr(model, "frozen_module_names", []):
            getattr(model, module_name).eval()

        total_loss = 0.0
        total_samples = 0
        for input_image, sequence, target_image, _, _, _ in train_loader:
            input_image = input_image.to(device, non_blocking=True)
            sequence = sequence.to(device, non_blocking=True)
            target_image = target_image.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(input_image, sequence), target_image)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(training_config["gradient_clip_norm"]),
            )
            optimizer.step()
            batch_size = input_image.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        train_loss = total_loss / total_samples
        val_loss = mean_image_loss(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        epoch_seconds = time.time() - started
        metrics = {
            "train_image_mse": train_loss,
            "val_image_mse": val_loss,
            "learning_rate": learning_rate,
            "epoch_seconds": epoch_seconds,
        }
        history.append({"epoch": epoch, **metrics})
        LOGGER.info(
            "Epoch %d/%d | train_mse=%.6f | val_mse=%.6f | lr=%.2e",
            epoch,
            int(stage_config["epochs"]),
            train_loss,
            val_loss,
            learning_rate,
        )
        if mlflow_module is not None:
            mlflow_module.log_metrics(metrics, step=epoch)

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_val_image_mse": best_loss,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(
                stage_config["early_stopping_patience"]
            ):
                LOGGER.info("Early stopping at epoch %d", epoch)
                break

    return pd.DataFrame(history)


def create_model(n_features, stage_config, model_config, device):
    return Multimodal(
        n_features=n_features,
        hidden_size=int(model_config["hidden_size"]),
        num_layers=int(model_config["num_layers"]),
        lstm_dropout=float(stage_config["lstm_dropout"]),
        feature_dropout=float(stage_config["feature_dropout"]),
    ).to(device)


def setup_mlflow(config):
    mlflow_config = config["mlflow"]
    if not bool(mlflow_config["enabled"]):
        return None
    load_dotenv(ENV_PATH)
    variable_name = mlflow_config["tracking_uri_environment_variable"]
    tracking_uri = os.getenv(variable_name)
    if not tracking_uri:
        raise RuntimeError(f"{variable_name} is missing from {ENV_PATH}")
    if "://" not in tracking_uri:
        tracking_uri = f"http://{tracking_uri}"
        LOGGER.warning(
            "%s has no URL scheme; using HTTP",
            variable_name,
        )
    if not tracking_uri.startswith(("http://", "https://")):
        raise ValueError(
            f"{variable_name} must be an HTTP or HTTPS URL"
        )
    import mlflow
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(mlflow_config["experiment_name"])
    return mlflow


def flatten_params(value, prefix=""):
    flattened = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            flattened.update(flatten_params(item, name))
        elif isinstance(item, (list, tuple)):
            flattened[name] = json.dumps(item)
        else:
            flattened[name] = item
    return flattened


def stage_output_directory(stage, config):
    root = project_path(config["artifacts"]["root"])
    key = "lab_directory" if stage == "lab" else "data4_directory"
    path = root / config["artifacts"][key]
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_stage(stage, config, device, mlflow_module):
    seed = int(config["training"]["seed"])
    set_seed(seed)
    stage_config = config["training"][stage]
    dataset_result, features, split = build_datasets(stage, config, seed)
    train_dataset, val_dataset, test_dataset, scaler = dataset_result
    loaders = create_loaders(
        (train_dataset, val_dataset, test_dataset),
        stage_config,
        config["training"],
        device,
        seed,
    )
    output_directory = stage_output_directory(stage, config)
    checkpoint_path = output_directory / "best_model.pth"
    model = create_model(
        len(features), stage_config, config["model"], device
    )

    if stage == "data4":
        lab_checkpoint = stage_output_directory("lab", config) / "best_model.pth"
        if not lab_checkpoint.is_file():
            raise FileNotFoundError(
                f"Data 4 fine-tuning requires {lab_checkpoint}"
            )
        checkpoint = torch.load(
            lab_checkpoint, map_location=device, weights_only=False
        )
        model = load_pretrained_multimodal(
            model,
            checkpoint,
            n_pretrained_features=len(LAB_FEATURES_COLS),
            freeze_visual=bool(
                config["model"]["freeze_visual_during_finetuning"]
            ),
        )

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(stage_config["learning_rate"]),
        weight_decay=float(stage_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(stage_config["scheduler_factor"]),
        patience=int(stage_config["scheduler_patience"]),
        min_lr=float(stage_config["minimum_learning_rate"]),
    )

    run_context = (
        mlflow_module.start_run(
            run_name=stage_config["run_name"],
            tags={"stage": stage, "model": config["model"]["name"]},
        )
        if mlflow_module is not None
        else nullcontext()
    )
    with run_context:
        if mlflow_module is not None:
            parameters = {
                **flatten_params(config["model"], "model"),
                **flatten_params(stage_config, "training"),
                "dataset": stage,
                "features": json.dumps(features),
                "train_groups": len(split["train"]),
                "validation_groups": len(split["val"]),
                "test_groups": len(split["test"]),
                "train_windows": len(train_dataset),
                "validation_windows": len(val_dataset),
                "test_windows": len(test_dataset),
                "device": str(device),
                "seed": seed,
            }
            mlflow_module.log_params(parameters)
            mlflow_module.log_artifact(
                str(CONFIG_PATH), artifact_path="configuration"
            )

        history = train_model(
            model,
            loaders[0],
            loaders[1],
            optimizer,
            scheduler,
            stage_config,
            config["training"],
            checkpoint_path,
            device,
            mlflow_module,
        )
        load_checkpoint(model, checkpoint_path, device)
        evaluation = evaluate_model(
            model, loaders[2], device, PREDICTION_LENGTH
        )

        history_path = output_directory / "history.csv"
        horizon_path = output_directory / "metrics_by_horizon.csv"
        scaler_path = output_directory / "scaler.joblib"
        history.to_csv(history_path, index=False)
        evaluation["metrics_by_horizon"].to_csv(horizon_path, index=False)
        joblib.dump(scaler, scaler_path)

        final_metrics = {
            key: value
            for key, value in evaluation.items()
            if key != "metrics_by_horizon"
        }
        LOGGER.info(
            "%s evaluation | image_mse=%.6f | mae=%.6f | rmse=%.6f",
            stage,
            final_metrics["image_mse"],
            final_metrics["forecast_mae"],
            final_metrics["forecast_rmse"],
        )
        if mlflow_module is not None:
            mlflow_module.log_metrics(final_metrics)
            mlflow_module.log_artifacts(
                str(output_directory), artifact_path="training_outputs"
            )
            if bool(config["mlflow"]["log_pytorch_model"]):
                mlflow_module.pytorch.log_model(model, name="model")

    return {"stage": stage, **final_metrics}


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    config = load_config()
    stages = list(config["training"]["stages"])
    unknown_stages = set(stages).difference({"lab", "data4"})
    if unknown_stages:
        raise ValueError(f"Unknown training stages: {sorted(unknown_stages)}")
    device = resolve_device(config["training"]["device"])
    mlflow_module = setup_mlflow(config)
    LOGGER.info("Device: %s", device)
    LOGGER.info("Training stages: %s", stages)

    results = [
        run_stage(stage, config, device, mlflow_module)
        for stage in stages
    ]
    summary = pd.DataFrame(results)
    summary_path = project_path(config["artifacts"]["root"]) / "summary.csv"
    summary.to_csv(summary_path, index=False)
    LOGGER.info("Saved training summary: %s", summary_path)


if __name__ == "__main__":
    main()
