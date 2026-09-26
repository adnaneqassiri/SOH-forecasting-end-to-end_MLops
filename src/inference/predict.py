#!/usr/bin/env python3
"""Forecast the next SOH values from one vehicle's raw telemetry CSV.

The CSV is long-form: every row is one telemetry sample and rows belonging to
the same charging event share ``charge_segment``. Required columns are::

    charge_segment, capacity, volt, current, soc, max_single_volt,
    min_single_volt, max_temp, min_temp, timestamp

``car`` or ``vehicle_id`` is optional, but the file must contain at most one
vehicle. ``mileage`` is also optional because it is not a model input.
Timestamps may be numeric seconds or values understood by pandas as datetimes.

Inference deliberately uses only the latest configured history window. It
does not calculate the offline EMD target (``SOH_ref``), which would use future
information and is unavailable in a real deployment.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

from src.data.data4_process import CycleAccumulator, SIGNAL_COLUMNS
from src.data.datasets import (
    DATA4_FEATURES_COLS,
    HISTORY_LENGTH,
    MASKED_FEATURE_TO_MASK,
    MASK_FEATURES_COLS,
    PREDICTION_LENGTH,
    image_to_soh,
    soh_to_image,
)
from src.model.model import Multimodal
from src.training.evaluate import load_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
LOGGER = logging.getLogger(__name__)

RAW_REQUIRED_COLUMNS = {
    "charge_segment",
    "capacity",
    *SIGNAL_COLUMNS,
}
RAW_NUMERIC_SIGNAL_COLUMNS = [
    column for column in SIGNAL_COLUMNS if column != "timestamp"
]


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the project configuration used to construct the trained model."""
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return config


def project_path(value: str | Path) -> Path:
    """Resolve a configuration path relative to the repository root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def default_artifact_paths(config: dict[str, Any]) -> tuple[Path, Path]:
    """Return the fine-tuned checkpoint and scaler paths."""
    directory = (
        project_path(config["artifacts"]["root"])
        / config["artifacts"]["data4_directory"]
    )
    return directory / "best_model.pth", directory / "scaler.joblib"


def resolve_device(name: str) -> torch.device:
    """Resolve ``auto`` and fail clearly when CUDA was explicitly requested."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def _normalize_identifier_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Accept the identifier names used by both raw and processed Data 4."""
    result = dataframe.copy()
    aliases = {
        "vehicle_id": "car",
        "charge_block_id": "charge_segment",
    }
    for alias, canonical in aliases.items():
        if alias in result.columns and canonical in result.columns:
            if not result[alias].equals(result[canonical]):
                raise ValueError(
                    f"Columns {alias!r} and {canonical!r} disagree"
                )
            result = result.drop(columns=alias)
        elif alias in result.columns:
            result = result.rename(columns={alias: canonical})
    return result


def _vehicle_label(dataframe: pd.DataFrame, fallback: str) -> str:
    if "car" not in dataframe.columns:
        return fallback
    identifiers = dataframe["car"].dropna().astype(str).unique()
    if len(identifiers) != 1:
        raise ValueError(
            "The input CSV must contain exactly one vehicle; found "
            f"{len(identifiers)} identifiers"
        )
    return str(identifiers[0])


def _as_numeric(
    dataframe: pd.DataFrame,
    columns: list[str],
    *,
    context: str,
) -> pd.DataFrame:
    result = dataframe.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    invalid = result[columns].isna().any(axis=1)
    if invalid.any():
        first_rows = result.index[invalid].tolist()[:5]
        raise ValueError(
            f"{context} contains missing or non-numeric values in rows "
            f"{first_rows}"
        )
    values = result[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{context} contains infinite values")
    return result


def _timestamp_seconds(values: pd.Series, charge_segment: int) -> np.ndarray:
    """Convert numeric or datetime timestamps to seconds for one event."""
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        seconds = numeric.to_numpy(dtype=np.float64)
    else:
        datetimes = pd.to_datetime(values, errors="coerce", utc=True)
        if datetimes.isna().any():
            raise ValueError(
                f"charge_segment {charge_segment} has invalid timestamps"
            )
        seconds = (
            (datetimes - datetimes.min()).dt.total_seconds().to_numpy(
                dtype=np.float64
            )
        )
    if not np.isfinite(seconds).all():
        raise ValueError(
            f"charge_segment {charge_segment} has non-finite timestamps"
        )
    return seconds


def _single_cycle_value(
    cycle: pd.DataFrame,
    column: str,
    charge_segment: int,
    *,
    default: float | None = None,
) -> float:
    """Read metadata that must be constant within a charging event."""
    if column not in cycle.columns:
        if default is None:
            raise ValueError(f"Missing required column: {column}")
        return default
    values = pd.to_numeric(cycle[column], errors="coerce").dropna().unique()
    if len(values) == 0:
        if default is not None:
            return default
        raise ValueError(
            f"charge_segment {charge_segment} has no {column} value"
        )
    if not np.isfinite(values).all():
        raise ValueError(
            f"charge_segment {charge_segment} has non-finite {column}"
        )
    if not np.allclose(values, values[0], rtol=1e-8, atol=1e-10):
        raise ValueError(
            f"{column} must be constant within charge_segment "
            f"{charge_segment}"
        )
    return float(values[0])


def estimate_nominal_capacity(capacities: pd.Series) -> float:
    """Estimate nameplate capacity exactly as the Data 4 preprocessing does."""
    numeric = pd.to_numeric(capacities, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric).all():
        raise ValueError("Capacity history contains missing or invalid values")
    if numeric.empty or numeric.le(0).any():
        raise ValueError("Capacity history must contain positive values")
    nominal_capacity = float(numeric.quantile(0.99))
    if not np.isfinite(nominal_capacity) or nominal_capacity <= 0:
        raise ValueError("Could not estimate a positive nominal capacity")
    return nominal_capacity


def extract_cycle_features(
    raw: pd.DataFrame,
    nominal_capacity: float | None = None,
) -> tuple[pd.DataFrame, float]:
    """Aggregate raw telemetry into the exact per-cycle training features."""
    raw = _normalize_identifier_columns(raw)
    missing = sorted(RAW_REQUIRED_COLUMNS.difference(raw.columns))
    if missing:
        raise ValueError(f"Missing required raw telemetry columns: {missing}")
    if raw.empty:
        raise ValueError("The input CSV is empty")

    raw = _as_numeric(
        raw,
        ["charge_segment", *RAW_NUMERIC_SIGNAL_COLUMNS],
        context="Raw telemetry",
    )
    segment_values = raw["charge_segment"].to_numpy(dtype=np.float64)
    if not np.equal(segment_values, np.floor(segment_values)).all():
        raise ValueError("charge_segment values must be integers")
    raw["charge_segment"] = segment_values.astype(np.int64)

    rows: list[dict[str, float | int]] = []
    for charge_segment, cycle in raw.groupby("charge_segment", sort=True):
        charge_segment = int(charge_segment)
        capacity = _single_cycle_value(cycle, "capacity", charge_segment)
        if capacity <= 0:
            raise ValueError(
                f"charge_segment {charge_segment} has non-positive capacity"
            )
        mileage = _single_cycle_value(
            cycle, "mileage", charge_segment, default=0.0
        )

        seconds = _timestamp_seconds(cycle["timestamp"], charge_segment)
        order = np.argsort(seconds, kind="stable")
        signal = cycle.iloc[order][RAW_NUMERIC_SIGNAL_COLUMNS].to_numpy(
            dtype=np.float64
        )
        signal = np.column_stack((signal, seconds[order]))

        accumulator = CycleAccumulator(
            car=0,
            charge_segment=charge_segment,
        )
        # A real CSV supplies one continuous event rather than the original
        # dataset's 128-row pickle snippets. Passing file_id=0 keeps timestamp
        # units and all feature formulas identical to the training extractor.
        accumulator.update(0, signal, capacity, mileage)
        quality = accumulator.finalize_quality()
        if not quality["usable"]:
            raise ValueError(
                f"charge_segment {charge_segment} is unusable; each event "
                "needs at least two finite telemetry samples"
            )
        rows.append(accumulator.feature_row())

    features = pd.DataFrame(rows).sort_values(
        "charge_segment", kind="stable"
    ).reset_index(drop=True)
    if features["charge_segment"].duplicated().any():
        raise ValueError("Duplicate charge_segment values after aggregation")

    if nominal_capacity is None:
        nominal_capacity = estimate_nominal_capacity(features["capacity"])
        LOGGER.info(
            "Estimated nominal capacity from observed history: %.6f",
            nominal_capacity,
        )
    nominal_capacity = float(nominal_capacity)
    if not np.isfinite(nominal_capacity) or nominal_capacity <= 0:
        raise ValueError("nominal_capacity must be a positive finite value")

    features["SOH_hist"] = features["capacity"] / nominal_capacity
    return features, nominal_capacity


def load_vehicle_csv(
    path: Path,
    nominal_capacity: float | None = None,
) -> tuple[pd.DataFrame, float, str]:
    """Read and process a one-vehicle long-form telemetry CSV."""
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    raw = pd.read_csv(path)
    raw = _normalize_identifier_columns(raw)
    vehicle = _vehicle_label(raw, fallback=path.stem)
    features, resolved_capacity = extract_cycle_features(
        raw, nominal_capacity=nominal_capacity
    )
    return features, resolved_capacity, vehicle


def apply_saved_scaler(
    features: pd.DataFrame,
    scaler: Any,
) -> pd.DataFrame:
    """Apply the mask-aware scaler serialized by Data 4 training."""
    if not isinstance(scaler, dict):
        raise ValueError(
            "The Data 4 scaler must be the mask-aware dictionary produced "
            "by src.training.train"
        )
    expected_features = list(scaler.get("feature_cols", []))
    if expected_features != DATA4_FEATURES_COLS:
        raise ValueError(
            "Scaler feature order is incompatible with this project: "
            f"{expected_features} != {DATA4_FEATURES_COLS}"
        )

    required = [*DATA4_FEATURES_COLS, *MASK_FEATURES_COLS]
    missing = sorted(set(required).difference(features.columns))
    if missing:
        raise ValueError(f"Cycle features are missing columns: {missing}")
    if features[required].isna().any().any():
        raise ValueError("Cycle features contain missing values")

    result = features.copy()
    result["soh_history_raw"] = result["SOH_hist"].astype(float)

    ordinary_features = list(scaler.get("ordinary_features", []))
    ordinary_scaler = scaler.get("ordinary_scaler")
    if ordinary_scaler is None or not ordinary_features:
        raise ValueError("Scaler is missing ordinary feature statistics")
    result.loc[:, ordinary_features] = ordinary_scaler.transform(
        result.loc[:, ordinary_features]
    )

    masked_stats = scaler.get("masked_feature_stats", {})
    expected_masked = {
        feature
        for feature in MASKED_FEATURE_TO_MASK
        if feature in DATA4_FEATURES_COLS
    }
    if set(masked_stats) != expected_masked:
        raise ValueError(
            "Scaler masked-feature statistics are incompatible with this "
            "project"
        )
    for feature, statistics in masked_stats.items():
        mask_column = statistics["mask_column"]
        masks = result[mask_column]
        if not set(masks.unique()).issubset({0, 1}):
            raise ValueError(f"{mask_column} must contain only 0 and 1")
        observed = masks.eq(1)
        scale = float(statistics["scale"])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid saved scale for {feature}")
        result.loc[observed, feature] = (
            result.loc[observed, feature] - float(statistics["mean"])
        ) / scale
        result.loc[~observed, feature] = 0.0

    model_values = result[DATA4_FEATURES_COLS].to_numpy(dtype=np.float32)
    if not np.isfinite(model_values).all():
        raise ValueError("Scaling produced non-finite model features")
    return result


def prepare_latest_window(
    features: pd.DataFrame,
    scaler: Any,
) -> tuple[torch.Tensor, torch.Tensor, float, pd.DataFrame]:
    """Create the image and numerical tensors for the latest history window."""
    if len(features) < HISTORY_LENGTH:
        raise ValueError(
            f"At least {HISTORY_LENGTH} charging events are required; "
            f"received {len(features)}"
        )
    ordered = features.sort_values("charge_segment", kind="stable")
    history = ordered.tail(HISTORY_LENGTH).copy().reset_index(drop=True)
    scaled_history = apply_saved_scaler(history, scaler)

    raw_soh = scaled_history["soh_history_raw"].to_numpy(dtype=np.float32)
    average_soh = float(raw_soh.mean())
    input_image = soh_to_image(raw_soh, avg=average_soh)
    sequence = scaled_history[DATA4_FEATURES_COLS].to_numpy(
        dtype=np.float32
    )

    image_tensor = torch.from_numpy(input_image).unsqueeze(0)
    sequence_tensor = torch.from_numpy(sequence).unsqueeze(0)
    return image_tensor, sequence_tensor, average_soh, history


def create_model(
    config: dict[str, Any],
    device: torch.device,
) -> Multimodal:
    """Construct the Data 4 architecture used during fine-tuning."""
    stage_config = config["training"]["data4"]
    model_config = config["model"]
    return Multimodal(
        n_features=len(DATA4_FEATURES_COLS),
        hidden_size=int(model_config["hidden_size"]),
        num_layers=int(model_config["num_layers"]),
        lstm_dropout=float(stage_config["lstm_dropout"]),
        feature_dropout=float(stage_config["feature_dropout"]),
    ).to(device)


@torch.no_grad()
def forecast_latest_window(
    model: torch.nn.Module,
    input_image: torch.Tensor,
    sequence: torch.Tensor,
    average_soh: float,
    device: torch.device,
    last_observed_soh: float | None = None,
) -> np.ndarray:
    """Run the model and decode the future part of its trajectory image."""
    model.eval()
    predicted_image = model(
        input_image.to(device),
        sequence.to(device),
    )[0]
    trajectory = image_to_soh(
        predicted_image,
        avg=average_soh,
        n_points=HISTORY_LENGTH + PREDICTION_LENGTH,
    )
    forecast = np.asarray(
        trajectory[-PREDICTION_LENGTH:], dtype=np.float64
    )
    if last_observed_soh is not None:
        last_observed_soh = float(last_observed_soh)
        if not np.isfinite(last_observed_soh):
            raise ValueError("last_observed_soh must be finite")
        forecast = last_observed_soh + (
            forecast - float(trajectory[HISTORY_LENGTH - 1])
        )
    if not np.isfinite(forecast).all():
        raise RuntimeError("The model produced non-finite SOH predictions")
    return forecast


def predict_vehicle(
    input_csv: Path,
    checkpoint_path: Path,
    scaler_path: Path,
    device: torch.device,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Run end-to-end latest-window inference for one raw vehicle CSV."""
    if not scaler_path.is_file():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    features, resolved_capacity, vehicle = load_vehicle_csv(input_csv)
    scaler = joblib.load(scaler_path)
    input_image, sequence, average_soh, history = prepare_latest_window(
        features, scaler
    )

    model = create_model(config, device)
    load_checkpoint(model, checkpoint_path, device)
    forecast = forecast_latest_window(
        model,
        input_image,
        sequence,
        average_soh,
        device,
        last_observed_soh=float(history.iloc[-1]["SOH_hist"]),
    )

    latest = history.iloc[-1]
    return pd.DataFrame(
        {
            "vehicle_id": [vehicle] * PREDICTION_LENGTH,
            "history_end_charge_segment": [
                int(latest["charge_segment"])
            ] * PREDICTION_LENGTH,
            "horizon": np.arange(1, PREDICTION_LENGTH + 1),
            "predicted_soh": forecast,
            "predicted_capacity": forecast * resolved_capacity,
        }
    )


def parse_args() -> argparse.Namespace:
    config = load_config()
    default_checkpoint, default_scaler = default_artifact_paths(config)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Long-form raw telemetry CSV for exactly one vehicle.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=default_checkpoint,
        help=f"Fine-tuned checkpoint (default: {default_checkpoint}).",
    )
    parser.add_argument(
        "--scaler",
        type=Path,
        default=default_scaler,
        help=f"Training scaler (default: {default_scaler}).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=config["training"]["device"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output CSV. Predictions are always printed to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    config = load_config()
    device = resolve_device(args.device)
    LOGGER.info("Device: %s", device)
    predictions = predict_vehicle(
        input_csv=args.input_csv,
        checkpoint_path=args.checkpoint,
        scaler_path=args.scaler,
        device=device,
        config=config,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(args.output, index=False)
        LOGGER.info("Saved predictions: %s", args.output)
    print(predictions.to_string(index=False))


if __name__ == "__main__":
    main()
