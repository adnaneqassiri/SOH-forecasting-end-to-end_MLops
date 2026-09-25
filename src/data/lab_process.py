"""Build one cleaned laboratory battery dataset from MATLAB files."""

from pathlib import Path
import logging
import h5py
import numpy as np
import pandas as pd
import yaml


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
SOC_INTERVALS = (
    (40, 50),
    (50, 60),
    (60, 70),
    (70, 80),
    (80, 90),
    (90, 100),
)

TIMESERIES_FIELDS = ("t", "I", "V", "T", "Qc", "Qd")
LOGGER = logging.getLogger(__name__)


def load_config() -> dict:
    """Load and validate the project configuration."""
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Expected a mapping in configuration file: {CONFIG_PATH}"
        )

    return config


config = load_config()

RAW_DIR = Path(config["raw"]["lab"])
FEATURES = config["features"]["lab"]
LAB_PROCESSING = config["processing"]["lab"]
NOMINAL_CAPACITY = float(LAB_PROCESSING["nominal_capacity"])
OUTLIER_THRESHOLD = float(LAB_PROCESSING["outlier_threshold"])
if NOMINAL_CAPACITY <= 0:
    raise ValueError("processing.lab.nominal_capacity must be positive")
if OUTLIER_THRESHOLD <= 0:
    raise ValueError("processing.lab.outlier_threshold must be positive")
OUTPUT_PATH = Path(config["processed"]["lab"])
OUTPUT_PATH = OUTPUT_PATH / "lab.csv"


# ============================================================
# MATLAB helpers
# ============================================================

def matlab_char(dataset: h5py.Dataset) -> str:
    """Convert MATLAB char array into a Python string."""
    values = np.asarray(dataset).reshape(-1, order="F")

    return "".join(
        chr(int(value))
        for value in values
        if value
    ).strip()


def dereference(
    file: h5py.File,
    references: h5py.Dataset,
    index: int,
):
    """Dereference a MATLAB HDF5 object."""
    flat_references = np.asarray(references).reshape(-1, order="F")

    if not 0 <= index < len(flat_references):
        raise IndexError(
            f"Reference index {index} is outside a dataset with "
            f"{len(flat_references)} entries"
        )

    return file[flat_references[index]]


def numeric_vector(dataset: h5py.Dataset) -> np.ndarray:
    """Convert MATLAB numeric array into a flat NumPy array."""

    matlab_empty = np.asarray(
        dataset.attrs.get("MATLAB_empty", 0)
    )

    if matlab_empty.any():
        return np.empty(0, dtype=float)

    return np.asarray(dataset, dtype=float).reshape(-1, order="F")


# ============================================================
# Charging segment
# ============================================================

def extract_charging_segment(
    cycle_df: pd.DataFrame,
) -> pd.DataFrame | None:
    """Extract the continuous charging part of a cycle."""

    cycle_df = cycle_df.sort_values(
        "sample_index"
    ).reset_index(drop=True)

    if cycle_df.empty:
        return None

    # --------------------------------------------------------
    # Find where charging starts
    # --------------------------------------------------------

    charge_started = (
        (cycle_df["I"] > 0)
        | (
            cycle_df["Qc"]
            > cycle_df["Qc"].iloc[0]
        )
    )

    if not charge_started.any():
        return None

    start = int(
        np.flatnonzero(
            charge_started.to_numpy()
        )[0]
    )

    # --------------------------------------------------------
    # Find where discharge starts
    # --------------------------------------------------------

    qd_increase = (
        cycle_df["Qd"]
        .diff()
        .fillna(0)
        > 1e-7
    )

    discharge_current = cycle_df["I"] < -0.1

    end_candidates = np.flatnonzero(
        (
            qd_increase
            | discharge_current
        ).to_numpy()
        & (
            np.arange(len(cycle_df))
            > start
        )
    )

    if len(end_candidates):
        end = int(end_candidates[0])
    else:
        end = len(cycle_df)

    segment = cycle_df.iloc[start:end].copy()

    if len(segment) < 2:
        return None

    return segment


# ============================================================
# SOC interpolation
# ============================================================

def interpolate_by_soc(
    soc: np.ndarray,
    time_s: np.ndarray,
    voltage: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a clean monotonic SOC axis for interpolation."""

    points = pd.DataFrame(
        {
            "soc": soc,
            "time_s": time_s,
            "voltage": voltage,
        }
    )

    points = (
        points
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .groupby(
            "soc",
            as_index=False,
            sort=True,
        )
        .agg(
            time_s=("time_s", "mean"),
            voltage=("voltage", "mean"),
        )
    )

    return (
        points["soc"].to_numpy(),
        points["time_s"].to_numpy(),
        points["voltage"].to_numpy(),
    )


# ============================================================
# Feature engineering
# ============================================================

def calculate_cycle_features(
    cycle_df: pd.DataFrame,
) -> dict[str, float] | None:
    """Calculate charging features for one cycle."""

    segment = extract_charging_segment(cycle_df)

    if segment is None:
        return None

    required = [
        "t",
        "I",
        "V",
        "T",
        "Qc",
    ]

    segment = (
        segment
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(subset=required)
    )

    if len(segment) < 2:
        return None

    # --------------------------------------------------------
    # Arrays
    # --------------------------------------------------------

    time_s = (
        segment["t"]
        .to_numpy(dtype=float)
        * 60
    )

    current = segment["I"].to_numpy(dtype=float)
    voltage = segment["V"].to_numpy(dtype=float)
    temperature = segment["T"].to_numpy(dtype=float)
    qc = segment["Qc"].to_numpy(dtype=float)

    # --------------------------------------------------------
    # Estimate SOC from charged capacity
    # --------------------------------------------------------

    qc_range = qc.max() - qc.min()

    if (
        not np.isfinite(qc_range)
        or qc_range <= 0
    ):
        return None

    soc = (
        (qc - qc.min())
        / qc_range
        * 100
    )

    soc_axis, time_axis, voltage_axis = (
        interpolate_by_soc(
            soc,
            time_s,
            voltage,
        )
    )

    if len(soc_axis) < 2:
        return None

    # --------------------------------------------------------
    # Statistical features
    # --------------------------------------------------------

    features = {
        "voltage_mean": float(
            np.mean(voltage)
        ),

        "voltage_std": float(
            np.std(
                voltage,
                ddof=1,
            )
        ),

        "current_mean": float(
            np.mean(current)
        ),

        "current_std": float(
            np.std(
                current,
                ddof=1,
            )
        ),

        "max_temp_mean": float(
            np.mean(temperature)
        ),

        "min_temp_std": float(
            np.std(
                temperature,
                ddof=1,
            )
        ),
    }

    # --------------------------------------------------------
    # Charging duration per SOC interval
    # --------------------------------------------------------

    for start_soc, end_soc in SOC_INTERVALS:

        name = f"t{start_soc}_{end_soc}"

        if (
            soc_axis[0] <= start_soc
            and soc_axis[-1] >= end_soc
        ):

            start_time = np.interp(
                start_soc,
                soc_axis,
                time_axis,
            )

            end_time = np.interp(
                end_soc,
                soc_axis,
                time_axis,
            )

            duration = float(
                end_time - start_time
            )

            features[name] = (
                duration
                if duration >= 0
                else np.nan
            )

        else:
            features[name] = np.nan

    # --------------------------------------------------------
    # Voltage slope
    # --------------------------------------------------------

    if np.ptp(time_s) > 0:

        features["voltage_slope"] = float(
            np.polyfit(
                time_s,
                voltage,
                1,
            )[0]
        )

    else:
        features["voltage_slope"] = np.nan

    # --------------------------------------------------------
    # Voltage at 80% SOC
    # --------------------------------------------------------

    if (
        soc_axis[0] <= 80
        and soc_axis[-1] >= 80
    ):

        features["max_cell_voltage_at_80"] = float(
            np.interp(
                80,
                soc_axis,
                voltage_axis,
            )
        )

    else:
        features["max_cell_voltage_at_80"] = np.nan

    return features


# ============================================================
# MATLAB cycle → DataFrame
# ============================================================

def load_cycle(
    file: h5py.File,
    cycles_group: h5py.Group,
    cycle_offset: int,
) -> pd.DataFrame:
    """Load one cycle from MATLAB."""

    arrays = {
        field: numeric_vector(
            dereference(
                file,
                cycles_group[field],
                cycle_offset,
            )
        )
        for field in TIMESERIES_FIELDS
    }

    sample_count = max(
        (
            len(values)
            for values in arrays.values()
        ),
        default=0,
    )

    data = {
        "sample_index": np.arange(
            1,
            sample_count + 1,
        )
    }

    for field, values in arrays.items():

        data[field] = (
            pd.Series(
                values,
                dtype=float,
            )
            .reindex(
                range(sample_count)
            )
        )

    return pd.DataFrame(data)


# ============================================================
# Cell processing
# ============================================================

def process_cell(
    file: h5py.File,
    batch: h5py.Group,
    batch_date: str,
    cell_offset: int,
    cell_index: int,
) -> list[dict]:
    """Process all cycles from one battery cell."""

    cell_id = (
        f"{batch_date}_"
        f"cell_{cell_offset + 1:03d}"
    )

    summary = dereference(
        file,
        batch["summary"],
        cell_offset,
    )

    capacities = numeric_vector(
        summary["QDischarge"]
    )

    cycles = dereference(
        file,
        batch["cycles"],
        cell_offset,
    )

    cycle_count = min(
        len(capacities),
        *(cycles[field].size for field in TIMESERIES_FIELDS),
    )

    if cycle_count != len(capacities):
        LOGGER.warning(
            "%s: summary contains %d cycles, but timeseries fields "
            "contain only %d complete cycles",
            cell_id,
            len(capacities),
            cycle_count,
        )

    rows = []

    for cycle_offset in range(cycle_count):

        capacity = capacities[cycle_offset]

        # ----------------------------------------------------
        # Basic capacity cleaning
        # ----------------------------------------------------

        if not np.isfinite(capacity):
            continue

        if capacity <= 0:
            continue

        if capacity > NOMINAL_CAPACITY:
            continue

        # ----------------------------------------------------
        # Load cycle
        # ----------------------------------------------------

        cycle_df = load_cycle(
            file,
            cycles,
            cycle_offset,
        )

        # ----------------------------------------------------
        # Feature engineering
        # ----------------------------------------------------

        features = calculate_cycle_features(
            cycle_df
        )

        if features is None:
            continue

        row = {
            "cell_index": cell_index,
            "cell_id": cell_id,
            "cycle_index": cycle_offset + 1,
            **features,
            "SOH_hist": float(
                capacity
                / NOMINAL_CAPACITY
            ),
        }

        rows.append(row)

    LOGGER.info(
        "%s: %d valid cycles",
        cell_id,
        len(rows),
    )

    return rows


# ============================================================
# Cleaning
# ============================================================

def clean_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Final dataset cleaning."""

    # Replace infinities
    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # --------------------------------------------------------
    # Invalid SOH
    # --------------------------------------------------------

    df = df[
        df["SOH_hist"].between(
            0,
            1,
            inclusive="both",
        )
    ]

    # --------------------------------------------------------
    # Remove duplicate cycles
    # --------------------------------------------------------

    df = df.drop_duplicates(
        subset=[
            "cell_id",
            "cycle_index",
        ]
    )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = df.sort_values(
        [
            "cell_index",
            "cycle_index",
        ]
    )

    local_median = df.groupby("cell_id")["SOH_hist"].transform(
        lambda values: values.rolling(
            window=5,
            center=True,
            min_periods=1,
        ).median()
    )
    outliers = (df["SOH_hist"] - local_median).abs() > OUTLIER_THRESHOLD
    if outliers.any():
        LOGGER.warning(
            "Dropping %d lab SOH outliers beyond %.4f of the local median",
            int(outliers.sum()),
            OUTLIER_THRESHOLD,
        )
        df = df.loc[~outliers]

    return df.reset_index(drop=True)


# ============================================================
# Full dataset
# ============================================================

def build_dataset() -> pd.DataFrame:
    """Read every MATLAB file and build one DataFrame."""

    mat_files = sorted(
        RAW_DIR.glob("*.mat")
    )

    if not mat_files:
        raise FileNotFoundError(
            f"No MATLAB files found in {RAW_DIR}"
        )

    rows = []

    cell_index = 0

    # ========================================================
    # MATLAB files
    # ========================================================

    for mat_path in mat_files:

        LOGGER.info(
            "Processing %s",
            mat_path.name,
        )

        with h5py.File(
            mat_path,
            "r",
        ) as file:

            batch = file["batch"]

            batch_date = matlab_char(
                file["batch_date"]
            )

            number_cells = batch["summary"].size

            # =================================================
            # Cells
            # =================================================

            for cell_offset in range(
                number_cells
            ):

                cell_index += 1

                cell_rows = process_cell(
                    file=file,
                    batch=batch,
                    batch_date=batch_date,
                    cell_offset=cell_offset,
                    cell_index=cell_index,
                )

                rows.extend(cell_rows)

    if not rows:
        raise RuntimeError(
            "No valid cycles were extracted."
        )

    df = pd.DataFrame(rows)

    LOGGER.info(
        "Extracted %d cycles from %d cells",
        len(df),
        df["cell_index"].nunique(),
    )

    # ========================================================
    # Clean
    # ========================================================

    df = clean_dataset(df)

    # ========================================================
    # Keep only features defined in config
    # + identifiers
    # ========================================================

    columns = [
        "cell_index",
        "cell_id",
        "cycle_index",
        *FEATURES,
    ]

    missing_columns = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Features defined in config.yaml "
            f"were not generated: {missing_columns}"
        )

    df = df[columns]

    return df


# ============================================================
# Main
# ============================================================

def main() -> None:

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    LOGGER.info(
        "Raw directory: %s",
        RAW_DIR,
    )

    # --------------------------------------------------------
    # Process everything
    # --------------------------------------------------------

    df = build_dataset()

    # --------------------------------------------------------
    # Save ONE CSV
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    LOGGER.info(
        "Saved dataset: %s",
        OUTPUT_PATH,
    )

    LOGGER.info(
        "Final shape: %s",
        df.shape,
    )

    LOGGER.info(
        "Cells: %d",
        df["cell_index"].nunique(),
    )


if __name__ == "__main__":
    main()