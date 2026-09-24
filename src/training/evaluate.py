"""Small evaluation helpers for multimodal SOH training."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.datasets import image_to_soh


def load_checkpoint(model, path, device):
    """Load a training checkpoint into a model."""
    checkpoint = torch.load(
        Path(path),
        map_location=device,
        weights_only=False,
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def evaluate_model(model, loader, device, prediction_length):
    """Evaluate image reconstruction and future SOH forecasts."""
    model.eval()
    image_losses = []
    predictions = []
    targets = []
    trajectory_length = loader.dataset.windows.iloc[0]["target_end"] + 1
    trajectory_length -= loader.dataset.windows.iloc[0]["start"]

    for input_image, sequence, target_image, future_soh, avg, _ in loader:
        input_image = input_image.to(device)
        sequence = sequence.to(device)
        target_image = target_image.to(device)
        predicted_image = model(input_image, sequence)
        image_losses.append(
            torch.nn.functional.mse_loss(
                predicted_image, target_image
            ).item()
        )

        for index in range(predicted_image.shape[0]):
            trajectory = image_to_soh(
                predicted_image[index],
                avg=avg[index].item(),
                n_points=int(trajectory_length),
            )
            predictions.append(trajectory[-prediction_length:])
            targets.append(future_soh[index].numpy())

    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    errors = predictions - targets
    horizon_metrics = pd.DataFrame(
        {
            "horizon": np.arange(1, prediction_length + 1),
            "mae": np.mean(np.abs(errors), axis=0),
            "rmse": np.sqrt(np.mean(np.square(errors), axis=0)),
        }
    )
    return {
        "image_mse": float(np.mean(image_losses)),
        "forecast_mae": float(np.mean(np.abs(errors))),
        "forecast_rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "metrics_by_horizon": horizon_metrics,
    }


__all__ = ["evaluate_model", "load_checkpoint"]
