"""Evaluate saved lab and Data 4 SOH checkpoints."""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.data.datasets import IMAGE_REPRESENTATION, image_to_soh


LOGGER = logging.getLogger(__name__)


def load_checkpoint(model, path, device):
    """Load a compatible training checkpoint into a model."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    representation = checkpoint.get("image_representation")
    if (
        representation is not None
        and representation != IMAGE_REPRESENTATION
    ):
        raise ValueError(
            f"{checkpoint_path} uses image representation "
            f"{representation!r}; expected {IMAGE_REPRESENTATION!r}"
        )

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    device,
    prediction_length,
    show_progress=False,
    description="Evaluate",
):
    """Evaluate image reconstruction and future SOH forecasts."""
    if len(loader.dataset) == 0:
        raise ValueError("The evaluation dataset is empty")

    model.eval()
    squared_image_error = 0.0
    image_element_count = 0
    predictions = []
    targets = []

    first_window = loader.dataset.windows.iloc[0]
    trajectory_length = int(
        first_window["target_end"] - first_window["start"] + 1
    )
    if trajectory_length <= int(prediction_length):
        raise ValueError(
            "Window trajectory length must exceed the prediction length"
        )

    progress = tqdm(
        loader,
        desc=description,
        unit="batch",
        leave=False,
        disable=not show_progress,
    )
    for input_image, sequence, target_image, future_soh, avg, _ in progress:
        input_image = input_image.to(device, non_blocking=True)
        sequence = sequence.to(device, non_blocking=True)
        target_image = target_image.to(device, non_blocking=True)
        predicted_image = model(input_image, sequence)

        squared_image_error += torch.nn.functional.mse_loss(
            predicted_image,
            target_image,
            reduction="sum",
        ).item()
        image_element_count += target_image.numel()

        for index in range(predicted_image.shape[0]):
            trajectory = image_to_soh(
                predicted_image[index],
                avg=avg[index].item(),
                n_points=trajectory_length,
            )
            predictions.append(trajectory[-prediction_length:])
            targets.append(
                future_soh[index].detach().cpu().numpy()
            )

        progress.set_postfix(
            image_mse=(
                f"{squared_image_error / image_element_count:.6f}"
            )
        )

    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if predictions.shape != targets.shape:
        raise ValueError(
            "Prediction and target shapes differ: "
            f"{predictions.shape} != {targets.shape}"
        )

    errors = predictions - targets
    horizon_metrics = pd.DataFrame(
        {
            "horizon": np.arange(1, prediction_length + 1),
            "mae": np.mean(np.abs(errors), axis=0),
            "rmse": np.sqrt(np.mean(np.square(errors), axis=0)),
        }
    )
    return {
        "image_mse": float(
            squared_image_error / image_element_count
        ),
        "forecast_mae": float(np.mean(np.abs(errors))),
        "forecast_rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "metrics_by_horizon": horizon_metrics,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate saved SOH model checkpoints on their test sets."
    )
    parser.add_argument(
        "--stage",
        choices=("all", "lab", "data4"),
        default="all",
        help="Checkpoint to evaluate (default: all).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        help="Override the device configured in config.yaml.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    return parser.parse_args()


def evaluate_stage(stage, config, device, show_progress):
    """Build one test split, load its checkpoint, and evaluate it."""
    # Imported lazily because train.py also imports the reusable functions above.
    from src.training.train import (
        build_datasets,
        create_loaders,
        create_model,
        stage_output_directory,
    )

    seed = int(config["training"]["seed"])
    dataset_result, features, split = build_datasets(stage, config, seed)
    train_dataset, val_dataset, test_dataset, _ = dataset_result
    loaders = create_loaders(
        (train_dataset, val_dataset, test_dataset),
        config["training"][stage],
        config["training"],
        device,
        seed,
    )

    output_directory = stage_output_directory(stage, config)
    checkpoint_path = output_directory / "best_model.pth"
    model = create_model(
        len(features),
        config["training"][stage],
        config["model"],
        device,
    )
    load_checkpoint(model, checkpoint_path, device)

    LOGGER.info(
        "Evaluating %s checkpoint on %d test windows from groups %s",
        stage,
        len(test_dataset),
        split["test"],
    )
    result = evaluate_model(
        model,
        loaders[2],
        device,
        int(config["sequence"]["prediction_length"]),
        show_progress=show_progress,
        description=f"Evaluate {stage}",
    )

    horizon_path = output_directory / "evaluation_metrics_by_horizon.csv"
    result["metrics_by_horizon"].to_csv(horizon_path, index=False)
    LOGGER.info("Saved horizon metrics: %s", horizon_path)
    return {
        "stage": stage,
        "test_groups": len(split["test"]),
        "test_windows": len(test_dataset),
        "image_mse": result["image_mse"],
        "forecast_mae": result["forecast_mae"],
        "forecast_rmse": result["forecast_rmse"],
    }


def main():
    from src.training.train import (
        load_config,
        project_path,
        resolve_device,
        set_seed,
    )

    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    config = load_config()
    set_seed(int(config["training"]["seed"]))
    device_name = args.device or config["training"]["device"]
    device = resolve_device(device_name)
    stages = ("lab", "data4") if args.stage == "all" else (args.stage,)

    LOGGER.info("Device: %s", device)
    results = [
        evaluate_stage(
            stage,
            config,
            device,
            show_progress=not args.no_progress,
        )
        for stage in stages
    ]
    summary = pd.DataFrame(results)
    summary_path = (
        project_path(config["artifacts"]["root"])
        / "evaluation_summary.csv"
    )
    summary.to_csv(summary_path, index=False)
    LOGGER.info("Saved evaluation summary: %s", summary_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()


__all__ = ["evaluate_model", "load_checkpoint"]
