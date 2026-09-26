"""PyTorch datasets for laboratory and Dataset-IV battery features."""

from collections.abc import Iterable
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.interpolate import CubicSpline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
LOGGER = logging.getLogger(__name__)


def load_config() -> dict:
    """Load and validate the project configuration."""
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {CONFIG_PATH}")
    return config


CONFIG = load_config()


def validate_required_columns(dataframe, required):
    missing = sorted(set(required).difference(dataframe.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_unique_rows(dataframe, columns):
    if dataframe.duplicated(columns).any():
        raise ValueError(f"Duplicate rows found for identifiers: {columns}")


def validate_split_ids(dataframe, id_column, splits):
    normalized = {
        name: {int(value) for value in values}
        for name, values in splits.items()
    }
    names = list(normalized)
    for index, left_name in enumerate(names):
        if not normalized[left_name]:
            raise ValueError(f"{left_name} split is empty")
        for right_name in names[index + 1:]:
            overlap = normalized[left_name] & normalized[right_name]
            if overlap:
                raise ValueError(
                    f"{left_name} and {right_name} splits overlap: {sorted(overlap)}"
                )
    available = {int(value) for value in dataframe[id_column].unique()}
    requested = set().union(*normalized.values())
    missing = sorted(requested - available)
    if missing:
        raise ValueError(f"Unknown {id_column} values in splits: {missing}")


def generate_windows(dataframe):
    rows = []
    total_length = HISTORY_LENGTH + PREDICTION_LENGTH
    for vehicle_id, vehicle in dataframe.groupby("vehicle_id", sort=False):
        for start in range(len(vehicle) - total_length + 1):
            rows.append(
                {
                    "vehicle_id": int(vehicle_id),
                    "start": start,
                    "history_end": start + HISTORY_LENGTH - 1,
                    "target_end": start + total_length - 1,
                }
            )
    return pd.DataFrame(
        rows,
        columns=["vehicle_id", "start", "history_end", "target_end"],
    )


def validate_window_splits(splits):
    for name, windows in splits.items():
        if windows.empty:
            raise ValueError(
                f"{name} split has no windows of length "
                f"{HISTORY_LENGTH + PREDICTION_LENGTH}"
            )


def scale(dataframe, train_windows, feature_cols, return_scaler=False):
    if train_windows.empty:
        raise ValueError("The training split has no valid windows")

    result = dataframe.copy()
    result["soh_history_raw"] = result["SOH_hist"].astype(float)
    train_ids = train_windows["vehicle_id"].unique()
    train_rows = result.loc[result["vehicle_id"].isin(train_ids)]
    if train_rows.empty:
        raise ValueError("Cannot fit feature scalers: no training rows")

    uses_availability_masks = any(
        mask in dataframe.columns for mask in MASKED_FEATURE_TO_MASK.values()
    )
    if not uses_availability_masks:
        scaler = StandardScaler().fit(train_rows[feature_cols])
        result.loc[:, feature_cols] = scaler.transform(result[feature_cols])
        if return_scaler:
            return result, scaler
        return result

    masked_features = [
        feature
        for feature in MASKED_FEATURE_TO_MASK
        if feature in feature_cols
    ]
    required_masks = [
        MASKED_FEATURE_TO_MASK[feature] for feature in masked_features
    ]
    missing_masks = set(required_masks).difference(dataframe.columns)
    if missing_masks:
        raise ValueError(
            f"Dataset is missing availability masks: {sorted(missing_masks)}"
        )

    for mask_column in required_masks:
        values = set(dataframe[mask_column].dropna().unique())
        if not values.issubset({0, 1}):
            raise ValueError(
                f"{mask_column} must be binary, found {sorted(values)}"
            )

    ordinary_features = [
        feature for feature in feature_cols if feature not in masked_features
    ]
    ordinary_scaler = StandardScaler().fit(train_rows[ordinary_features])
    result.loc[:, ordinary_features] = ordinary_scaler.transform(
        result[ordinary_features]
    )

    masked_feature_stats = {}
    for feature in masked_features:
        mask_column = MASKED_FEATURE_TO_MASK[feature]
        train_observed = train_rows.loc[
            train_rows[mask_column].eq(1), feature
        ]
        if train_observed.empty:
            raise ValueError(
                f"Cannot scale {feature}: no observed value in training vehicles"
            )

        mean = float(train_observed.mean())
        scale_value = float(train_observed.std(ddof=0))
        if not np.isfinite(scale_value) or scale_value == 0:
            scale_value = 1.0

        observed = dataframe[mask_column].eq(1)
        result.loc[observed, feature] = (
            dataframe.loc[observed, feature] - mean
        ) / scale_value
        result.loc[~observed, feature] = 0.0
        masked_feature_stats[feature] = {
            "mask_column": mask_column,
            "mean": mean,
            "scale": scale_value,
        }

    scaler = {
        "feature_cols": list(feature_cols),
        "ordinary_features": ordinary_features,
        "ordinary_scaler": ordinary_scaler,
        "masked_feature_stats": masked_feature_stats,
        "mask_columns": [],
    }
    if return_scaler:
        return result, scaler
    return result


REAL_EV_FEATURES = list(CONFIG["features"]["real_ev"])
MASK_FEATURES_COLS = [
    column for column in REAL_EV_FEATURES if column.startswith("mask_")
]
MASKED_FEATURE_TO_MASK = {
    "t40_50": "mask_t40_50",
    "t50_60": "mask_t50_60",
    "t60_70": "mask_t60_70",
    "t70_80": "mask_t70_80",
    "t80_90": "mask_t80_90",
    "t90_100": "mask_t90_100",
    "max_cell_voltage_at_80": "mask_v80",
}
DATA4_FEATURES_COLS = [
    column for column in REAL_EV_FEATURES if column not in MASK_FEATURES_COLS
]
LAB_FEATURES_COLS = list(CONFIG["features"]["lab"])
DATA4_FEATURES = Path(CONFIG["processed"]["real_ev"]) / "real_ev.csv"
LAB_FEATURES = Path(CONFIG["processed"]["lab"]) / "lab.csv"
HISTORY_LENGTH = int(CONFIG["sequence"]["history_length"])
PREDICTION_LENGTH = int(CONFIG["sequence"]["prediction_length"])
if HISTORY_LENGTH <= 0 or PREDICTION_LENGTH <= 0:
    raise ValueError("Sequence lengths in config.yaml must be positive")
if DATA4_FEATURES_COLS[:len(LAB_FEATURES_COLS)] != LAB_FEATURES_COLS:
    raise ValueError(
        "Data 4 features must begin with the lab features in the same order"
    )
IMAGE_SIZE = int(CONFIG["model"]["image_size"])
MODEL_IMAGE_SHAPE = (IMAGE_SIZE, IMAGE_SIZE)
SOH_IMAGE_HALF_RANGE = 0.08
IMAGE_REPRESENTATION = "interpolated_curve_argmin_spline_v2"
DATA4_ID_COLUMNS = {"car": "vehicle_id", "charge_segment": "charge_block_id"}


def soh_to_image(values, avg=None):
    """Rasterize SOH using the original common 110-point coordinate system."""
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError("SOH trajectory must contain at least two finite values")
    total_length = HISTORY_LENGTH + PREDICTION_LENGTH
    if values.size > total_length:
        raise ValueError(
            f"SOH trajectory has {values.size} points; expected at most "
            f"{total_length}"
        )
    if avg is None:
        avg = float(values.mean())
    if not np.isfinite(avg):
        raise ValueError("Average SOH must be finite")

    cycle_positions = np.linspace(0, IMAGE_SIZE - 1, total_length)
    known_positions = cycle_positions[: values.size]
    last_known_column = int(np.floor(known_positions[-1]))
    x_pixels = np.arange(last_known_column + 1)
    interpolated = np.interp(x_pixels, known_positions, values)

    y_pixels = IMAGE_SIZE * (
        1.0
        - (
            interpolated
            - float(avg)
            + SOH_IMAGE_HALF_RANGE
        )
        / (2.0 * SOH_IMAGE_HALF_RANGE)
    )
    y_pixels = np.clip(
        np.rint(y_pixels), 0, IMAGE_SIZE - 1
    ).astype(np.int32)

    image = np.ones(MODEL_IMAGE_SHAPE, dtype=np.float32)
    image[y_pixels, x_pixels] = 0.0
    return image


def image_to_soh(image, avg, n_points):
    """Decode a rasterized SOH curve with column argmin and spline sampling."""
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.shape != MODEL_IMAGE_SHAPE:
        raise ValueError(
            f"SOH image must have shape {MODEL_IMAGE_SHAPE}; received {image.shape}"
        )
    if not np.isfinite(image).all() or not np.isfinite(avg):
        raise ValueError("SOH image and average must contain only finite values")
    n_points = int(n_points)
    if n_points < 2:
        raise ValueError("Decoded SOH trajectory must contain at least two points")

    height, width = MODEL_IMAGE_SHAPE
    pixel_rows = np.argmin(image, axis=0)
    pixel_soh = (
        (1.0 - pixel_rows / height) * (2.0 * SOH_IMAGE_HALF_RANGE)
        + float(avg)
        - SOH_IMAGE_HALF_RANGE
    )
    pixel_axis = np.arange(width, dtype=np.float32)
    output_axis = np.linspace(0.0, width - 1, n_points)
    return CubicSpline(pixel_axis, pixel_soh)(output_axis)


def _as_vehicle_ids(value):
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        return [int(value)]
    return [int(vehicle_id) for vehicle_id in value]


def load_data4_features(path=DATA4_FEATURES):
    """Load Data4 and normalize its identifier columns."""
    dataframe = pd.read_csv(path).rename(columns=DATA4_ID_COLUMNS)
    required = {
        "vehicle_id",
        "charge_block_id",
        *DATA4_FEATURES_COLS,
        *MASK_FEATURES_COLS,
        "SOH_ref",
    }
    validate_required_columns(dataframe, required)
    dataframe = dataframe[
        [
            "vehicle_id",
            "charge_block_id",
            *DATA4_FEATURES_COLS,
            *MASK_FEATURES_COLS,
            "SOH_ref",
        ]
    ].copy()
    dataframe["vehicle_id"] = dataframe["vehicle_id"].astype(int)
    dataframe["charge_block_id"] = dataframe["charge_block_id"].astype(int)
    dataframe = dataframe.sort_values(
        ["vehicle_id", "charge_block_id"]
    ).reset_index(drop=True)
    if dataframe.isna().any().any():
        raise ValueError("Data4 contains missing values in model columns")
    validate_unique_rows(dataframe, ["vehicle_id", "charge_block_id"])
    return dataframe


class Data4MultimodalDataset(Dataset):
    def __init__(
        self,
        windows,
        vehicles,
        feature_cols,
        soh_to_image,
    ):
        self.windows = windows.reset_index(drop=True)
        self.feature_cols = list(feature_cols)
        self.soh_to_image = soh_to_image
        self.vehicles = {
            int(vehicle_id): vehicle.sort_values(
                "charge_block_id"
            ).reset_index(drop=True)
            for vehicle_id, vehicle in vehicles.groupby("vehicle_id")
        }

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows.iloc[idx]
        vehicle_id = int(window["vehicle_id"])
        start = int(window["start"])
        history_end = int(window["history_end"])
        target_end = int(window["target_end"])
        vehicle = self.vehicles[vehicle_id]

        sequence = vehicle.iloc[
            start : history_end + 1
        ][self.feature_cols].to_numpy(dtype=np.float32)

        noisy_history = vehicle.iloc[
            start : history_end + 1
        ]["soh_history_raw"].to_numpy(dtype=np.float32)

        clean_trajectory = vehicle.iloc[
            start : target_end + 1
        ]["SOH_ref"].to_numpy(dtype=np.float32)

        future_soh = clean_trajectory[HISTORY_LENGTH:]
        avg = float(np.mean(noisy_history))

        input_image = np.asarray(
            self.soh_to_image(noisy_history, avg=avg),
            dtype=np.float32,
        )
        target_image = np.asarray(
            self.soh_to_image(clean_trajectory, avg=avg),
            dtype=np.float32,
        )
        if input_image.shape != MODEL_IMAGE_SHAPE:
            raise ValueError(
                f"Input SOH image must have shape {MODEL_IMAGE_SHAPE}; "
                f"received {input_image.shape}"
            )
        if target_image.shape != MODEL_IMAGE_SHAPE:
            raise ValueError(
                f"Target SOH image must have shape {MODEL_IMAGE_SHAPE}; "
                f"received {target_image.shape}"
            )
        if not np.isfinite(input_image).all() or not np.isfinite(target_image).all():
            raise ValueError("SOH image conversion produced NaN or infinite values")

        last_soh = float(noisy_history[-1])

        return (
            torch.tensor(input_image, dtype=torch.float32),
            torch.tensor(sequence, dtype=torch.float32),
            torch.tensor(target_image, dtype=torch.float32),
            torch.tensor(future_soh, dtype=torch.float32),
            torch.tensor(avg, dtype=torch.float32),
            torch.tensor(last_soh, dtype=torch.float32),
        )


def create_data4_multimodal_datasets(
    train_ids,
    val_id,
    test_id,
    soh_to_image,
    feature_cols=DATA4_FEATURES_COLS,
    return_scaler=False,
):
    feature_cols = list(feature_cols)
    train_ids = _as_vehicle_ids(train_ids)
    val_ids = _as_vehicle_ids(val_id)
    test_ids = _as_vehicle_ids(test_id)

    data4_df = load_data4_features()
    validate_split_ids(
        data4_df,
        "vehicle_id",
        {"train": train_ids, "val": val_ids, "test": test_ids},
    )
    windows = generate_windows(data4_df)

    train_windows = windows[windows["vehicle_id"].isin(train_ids)]
    val_windows = windows[windows["vehicle_id"].isin(val_ids)]
    test_windows = windows[windows["vehicle_id"].isin(test_ids)]
    validate_window_splits(
        {
            "train": train_windows,
            "val": val_windows,
            "test": test_windows,
        }
    )

    scaled = scale(
        data4_df,
        train_windows,
        feature_cols,
        return_scaler=return_scaler,
    )
    if return_scaler:
        data4_df_scaled, data4_scaler = scaled
    else:
        data4_df_scaled = scaled

    datasets = (
        Data4MultimodalDataset(
            train_windows, data4_df_scaled, feature_cols, soh_to_image
        ),
        Data4MultimodalDataset(
            val_windows, data4_df_scaled, feature_cols, soh_to_image
        ),
        Data4MultimodalDataset(
            test_windows, data4_df_scaled, feature_cols, soh_to_image
        ),
    )
    LOGGER.info(
        "Created Data 4 datasets: train=%d, val=%d, test=%d windows",
        *(len(dataset) for dataset in datasets),
    )
    if return_scaler:
        return (*datasets, data4_scaler)
    return datasets


LAB_ID_COLUMNS = {
    "cell_index": "vehicle_id",
    "cycle_index": "charge_block_id",
}


def load_lab_features(path=LAB_FEATURES):
    """Load lab features and normalize them to the shared model schema."""
    dataframe = pd.read_csv(path).rename(columns=LAB_ID_COLUMNS)
    required = {
        "vehicle_id",
        "charge_block_id",
        *LAB_FEATURES_COLS,
        "soh_clean",
    }
    validate_required_columns(dataframe, required)
    dataframe = dataframe[
        [
            "vehicle_id",
            "charge_block_id",
            *LAB_FEATURES_COLS,
            "soh_clean",
        ]
    ].copy()

    if dataframe.isna().any().any():
        raise ValueError(
            "LAB data contains missing model features; rerun LAB preprocessing"
        )

    dataframe["vehicle_id"] = dataframe["vehicle_id"].astype(int)
    dataframe["charge_block_id"] = dataframe["charge_block_id"].astype(int)
    dataframe["SOH_ref"] = dataframe["soh_clean"].astype(float)
    dataframe = dataframe.sort_values(
        ["vehicle_id", "charge_block_id"]
    ).reset_index(drop=True)
    validate_unique_rows(dataframe, ["vehicle_id", "charge_block_id"])
    return dataframe


def add_lab_noise(signal, sigma=0.03, rng=None):
    """Add the original LAB image-history augmentation."""
    signal = np.asarray(signal, dtype=np.float32)
    if rng is None:
        noise = np.random.normal(0.0, sigma, size=signal.shape)
    else:
        noise = rng.normal(0.0, sigma, size=signal.shape)
    return (signal + noise).astype(np.float32)


class LabMultimodalDataset(Data4MultimodalDataset):
    """LAB windows with noisy image history and clean image/forecast targets."""

    def __init__(
        self,
        windows,
        vehicles,
        feature_cols,
        soh_to_image,
        noise_seed,
    ):
        super().__init__(windows, vehicles, feature_cols, soh_to_image)
        self.noise_seed = noise_seed

    def __getitem__(self, idx):
        window = self.windows.iloc[idx]
        vehicle_id = int(window["vehicle_id"])
        start = int(window["start"])
        history_end = int(window["history_end"])
        target_end = int(window["target_end"])
        vehicle = self.vehicles[vehicle_id]

        sequence = vehicle.iloc[
            start : history_end + 1
        ][self.feature_cols].to_numpy(dtype=np.float32)
        clean_trajectory = vehicle.iloc[
            start : target_end + 1
        ]["SOH_ref"].to_numpy(dtype=np.float32)
        clean_history = clean_trajectory[:HISTORY_LENGTH]
        future_soh = clean_trajectory[HISTORY_LENGTH:]

        rng = (
            None
            if self.noise_seed is None
            else np.random.default_rng(self.noise_seed + idx)
        )
        noisy_history = add_lab_noise(clean_history.copy(), rng=rng)
        avg = float(np.mean(clean_history))

        input_image = np.asarray(
            self.soh_to_image(noisy_history, avg=avg), dtype=np.float32
        )
        target_image = np.asarray(
            self.soh_to_image(clean_trajectory, avg=avg), dtype=np.float32
        )
        last_soh = float(vehicle.iloc[history_end]["soh_history_raw"])

        return (
            torch.tensor(input_image, dtype=torch.float32),
            torch.tensor(sequence, dtype=torch.float32),
            torch.tensor(target_image, dtype=torch.float32),
            torch.tensor(future_soh, dtype=torch.float32),
            torch.tensor(avg, dtype=torch.float32),
            torch.tensor(last_soh, dtype=torch.float32),
        )


def create_lab_multimodal_datasets(
    train_ids,
    val_id,
    test_id,
    soh_to_image,
    feature_cols=LAB_FEATURES_COLS,
    return_scaler=False,
):
    """Create group-disjoint train, validation, and test lab datasets."""
    feature_cols = list(feature_cols)
    train_ids = _as_vehicle_ids(train_ids)
    val_ids = _as_vehicle_ids(val_id)
    test_ids = _as_vehicle_ids(test_id)

    lab_df = load_lab_features()
    validate_split_ids(
        lab_df,
        "vehicle_id",
        {"train": train_ids, "val": val_ids, "test": test_ids},
    )
    windows = generate_windows(lab_df)

    train_windows = windows[windows["vehicle_id"].isin(train_ids)]
    val_windows = windows[windows["vehicle_id"].isin(val_ids)]
    test_windows = windows[windows["vehicle_id"].isin(test_ids)]
    validate_window_splits(
        {
            "train": train_windows,
            "val": val_windows,
            "test": test_windows,
        }
    )

    scaled = scale(
        lab_df,
        train_windows,
        feature_cols,
        return_scaler=return_scaler,
    )
    if return_scaler:
        lab_df_scaled, lab_scaler = scaled
    else:
        lab_df_scaled = scaled

    datasets = (
        LabMultimodalDataset(
            train_windows,
            lab_df_scaled,
            feature_cols,
            soh_to_image,
            noise_seed=None,
        ),
        LabMultimodalDataset(
            val_windows,
            lab_df_scaled,
            feature_cols,
            soh_to_image,
            noise_seed=2026,
        ),
        LabMultimodalDataset(
            test_windows,
            lab_df_scaled,
            feature_cols,
            soh_to_image,
            noise_seed=2027,
        ),
    )
    LOGGER.info(
        "Created lab datasets: train=%d, val=%d, test=%d windows",
        *(len(dataset) for dataset in datasets),
    )
    if return_scaler:
        return (*datasets, lab_scaler)
    return datasets
