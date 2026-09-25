#!/usr/bin/env python3
"""Select manufacturer-1 Dataset-IV vehicles and build their feature CSV.

Only ``battery_dataset1.tar.gz`` is streamed; the other manufacturer archives
are deliberately excluded. A row in the output represents one labelled
``(car, charge_segment)``. Vehicle
selection favours the largest number of K+Q model windows, followed by signal
quality, SOC-feature availability, and the amount of telemetry per cycle.

No scaling or model-window materialisation is performed by this script.
"""

from __future__ import annotations

import argparse
import logging
import math
import pickle
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
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


CONFIG = load_config()
DATA4_PROCESSING = CONFIG["processing"]["data4"]


REAL_EV_FEATURES = list(CONFIG["features"]["real_ev"])
MASK_FEATURES_COLS = [
    feature for feature in REAL_EV_FEATURES if feature.startswith("mask_")
]
EV_BASE_FEATURES_COLS = [
    feature for feature in REAL_EV_FEATURES
    if feature not in MASK_FEATURES_COLS
]


DEFAULT_INPUT_DIR = Path(CONFIG["raw"]["real_ev"])
DEFAULT_OUTPUT_DIR = Path(CONFIG["processed"]["real_ev"])
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "real_ev.csv"
DEFAULT_REPORT = DEFAULT_OUTPUT_DIR / "vehicle_selection.csv"
MANUFACTURER_ID = 1
ARCHIVE_NAME = f"battery_dataset{MANUFACTURER_ID}.tar.gz"

SIGNAL_COLUMNS = (
    "volt",
    "current",
    "soc",
    "max_single_volt",
    "min_single_volt",
    "max_temp",
    "min_temp",
    "timestamp",
)

SOC_INTERVALS = ((40.0, 50.0), (50.0, 60.0), (60.0, 70.0),
                 (70.0, 80.0), (80.0, 90.0), (90.0, 99.0))
SOC_THRESHOLDS = (40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 99.0)

BASE_OUTPUT_COLUMNS = [
    "car",
    "charge_segment",
    "capacity",
    *EV_BASE_FEATURES_COLS,
    "mileage",
    "SOH_ref",
]

MASK_COLUMNS = MASK_FEATURES_COLS
OUTPUT_COLUMNS = [*BASE_OUTPUT_COLUMNS, *MASK_COLUMNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--top-n",
        type=int,
        default=int(DATA4_PROCESSING["top_n"]),
    )
    parser.add_argument("--history-length", type=int, default=100)
    parser.add_argument("--prediction-length", type=int, default=10)
    parser.add_argument(
        "--max-imf",
        type=int,
        default=int(DATA4_PROCESSING["max_imf"]),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50_000,
        help="Print progress after this many pickle snippets (0 disables it).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max-files-per-archive",
        type=int,
        default=None,
        help="Development/smoke-test limit for the manufacturer-1 archive.",
    )
    return parser.parse_args()


def load_ev_tuple(stream: BinaryIO) -> tuple[np.ndarray, dict]:
    """Load the same tuple format accepted by the EDA notebook."""
    while True:
        try:
            obj = pickle.load(stream)
        except EOFError as error:
            raise ValueError("No EV battery tuple found") from error
        if (
            isinstance(obj, tuple)
            and len(obj) == 2
            and isinstance(obj[0], np.ndarray)
            and isinstance(obj[1], dict)
        ):
            return obj[0], obj[1]


def iter_archive_pickles(
    archive: Path, max_files: int | None = None
) -> Iterator[tuple[int, np.ndarray, dict]]:
    """Yield pickle contents from a compressed tar without extracting files."""
    count = 0
    with tarfile.open(archive, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".pkl"):
                continue
            stem = Path(member.name).stem
            if not stem.isdigit():
                # The public README mentions column.pkl; it is not a sample.
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            signal, metadata = load_ev_tuple(extracted)
            yield int(stem), signal, metadata
            count += 1
            if max_files is not None and count >= max_files:
                return


@dataclass(slots=True)
class CycleAccumulator:
    car: int
    charge_segment: int
    n_rows: int = 0
    n_snippets: int = 0
    file_min: int = 2**63 - 1
    file_max: int = -1
    capacity_min: float = math.inf
    capacity_max: float = -math.inf
    mileage_min: float = math.inf
    mileage_max: float = -math.inf
    all_finite: bool = True
    soc_min: float = math.inf
    soc_max: float = -math.inf
    negative_soc_steps: int = 0
    soc_step_count: int = 0
    max_soc_drop: float = 0.0
    # volt, current, maxV, minV, maxT, minT, voltage spread, temp spread
    value_sum: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float64))
    value_sumsq: np.ndarray = field(
        default_factory=lambda: np.zeros(8, dtype=np.float64)
    )
    x_sum: float = 0.0
    x_sumsq: float = 0.0
    xv_sum: float = 0.0
    first_threshold_time: np.ndarray = field(
        default_factory=lambda: np.full(len(SOC_THRESHOLDS), np.inf, dtype=np.float64)
    )
    v80_lower_soc: float = -math.inf
    v80_lower_time: float = -math.inf
    v80_lower_value: float = math.nan
    v80_upper_soc: float = math.inf
    v80_upper_time: float = math.inf
    v80_upper_value: float = math.nan
    endpoints: list[tuple[int, float, float]] = field(default_factory=list)

    def update(
        self,
        file_id: int,
        signal: np.ndarray,
        capacity: float,
        mileage: float,
    ) -> None:
        array = np.asarray(signal, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != len(SIGNAL_COLUMNS) or len(array) == 0:
            self.all_finite = False
            return

        self.n_snippets += 1
        self.file_min = min(self.file_min, file_id)
        self.file_max = max(self.file_max, file_id)
        self.capacity_min = min(self.capacity_min, capacity)
        self.capacity_max = max(self.capacity_max, capacity)
        self.mileage_min = min(self.mileage_min, mileage)
        self.mileage_max = max(self.mileage_max, mileage)

        if not np.isfinite(array).all():
            self.all_finite = False
            return

        n = len(array)
        self.n_rows += n
        volt = array[:, 0]
        current = array[:, 1]
        soc = array[:, 2]
        max_volt = array[:, 3]
        min_volt = array[:, 4]
        max_temp = array[:, 5]
        min_temp = array[:, 6]
        timestamp = array[:, 7]
        values = np.column_stack(
            (
                volt,
                current,
                max_volt,
                min_volt,
                max_temp,
                min_temp,
                max_volt - min_volt,
                max_temp - min_temp,
            )
        )
        self.value_sum += values.sum(axis=0)
        self.value_sumsq += np.square(values).sum(axis=0)

        # All public snippets contain 128 samples at 10 s.  Using a file-based
        # absolute origin makes the slope and interval differences identical to
        # concatenating contiguous snippets and rebuilding session_time.
        absolute_time = file_id * 1280.0 + timestamp
        self.x_sum += float(absolute_time.sum())
        self.x_sumsq += float(np.square(absolute_time).sum())
        self.xv_sum += float(np.dot(absolute_time, volt))

        self.soc_min = min(self.soc_min, float(soc.min()))
        self.soc_max = max(self.soc_max, float(soc.max()))
        soc_diffs = np.diff(soc)
        self.negative_soc_steps += int(np.count_nonzero(soc_diffs < 0))
        self.soc_step_count += len(soc_diffs)
        if len(soc_diffs):
            self.max_soc_drop = max(
                self.max_soc_drop,
                float(max(0.0, -soc_diffs.min())),
            )
        self.endpoints.append((file_id, float(soc[0]), float(soc[-1])))

        for index, threshold in enumerate(SOC_THRESHOLDS):
            matches = np.flatnonzero(soc >= threshold)
            if matches.size:
                first_time = float(absolute_time[matches[0]])
                self.first_threshold_time[index] = min(
                    self.first_threshold_time[index], first_time
                )

        below = np.flatnonzero(soc <= 80.0)
        if below.size:
            best_soc = float(soc[below].max())
            candidates = below[soc[below] == best_soc]
            best_index = int(candidates[-1])
            best_time = float(absolute_time[best_index])
            if (
                best_soc > self.v80_lower_soc
                or (best_soc == self.v80_lower_soc and best_time > self.v80_lower_time)
            ):
                self.v80_lower_soc = best_soc
                self.v80_lower_time = best_time
                self.v80_lower_value = float(max_volt[best_index])

        above = np.flatnonzero(soc >= 80.0)
        if above.size:
            best_soc = float(soc[above].min())
            candidates = above[soc[above] == best_soc]
            best_index = int(candidates[0])
            best_time = float(absolute_time[best_index])
            if (
                best_soc < self.v80_upper_soc
                or (best_soc == self.v80_upper_soc and best_time < self.v80_upper_time)
            ):
                self.v80_upper_soc = best_soc
                self.v80_upper_time = best_time
                self.v80_upper_value = float(max_volt[best_index])

    def finalize_quality(self) -> dict[str, float | int | bool]:
        endpoints = sorted(self.endpoints)
        boundary_diffs = np.array(
            [endpoints[i][1] - endpoints[i - 1][2] for i in range(1, len(endpoints))],
            dtype=np.float64,
        )
        boundary_negative = int(np.count_nonzero(boundary_diffs < 0))
        max_boundary_drop = (
            float(max(0.0, -boundary_diffs.min())) if boundary_diffs.size else 0.0
        )
        step_count = self.soc_step_count + len(boundary_diffs)
        negative_count = self.negative_soc_steps + boundary_negative
        monotonic_fraction = 1.0 - negative_count / max(step_count, 1)
        contiguous = self.file_max - self.file_min + 1 == self.n_snippets
        metadata_consistent = math.isclose(
            self.capacity_min, self.capacity_max, rel_tol=1e-10, abs_tol=1e-10
        ) and math.isclose(
            self.mileage_min, self.mileage_max, rel_tol=1e-10, abs_tol=1e-8
        )
        usable = bool(
            self.all_finite
            and self.n_rows >= 2
            and contiguous
            and metadata_consistent
            and self.capacity_max > 0
        )
        return {
            "usable": usable,
            "contiguous": contiguous,
            "metadata_consistent": metadata_consistent,
            "monotonic_fraction": monotonic_fraction,
            "max_soc_drop": max(self.max_soc_drop, max_boundary_drop),
        }

    @staticmethod
    def _sample_std(total: float, total_sq: float, n: int) -> float:
        variance = (total_sq - total * total / n) / (n - 1)
        return math.sqrt(max(0.0, variance))

    def feature_row(self) -> dict[str, float | int]:
        means = self.value_sum / self.n_rows
        stds = np.array(
            [self._sample_std(self.value_sum[i], self.value_sumsq[i], self.n_rows)
             for i in range(len(self.value_sum))]
        )

        interval_values: list[float] = []
        interval_masks: list[int] = []
        threshold_index = {value: i for i, value in enumerate(SOC_THRESHOLDS)}
        for low, high in SOC_INTERVALS:
            observed = self.soc_min <= low and self.soc_max >= high
            low_time = self.first_threshold_time[threshold_index[low]]
            high_time = self.first_threshold_time[threshold_index[high]]
            # A zero duration means the sequence jumped across the whole SOC
            # band at one sample boundary; that interval was not actually
            # observed and must therefore remain masked.
            observed = (
                observed
                and np.isfinite(low_time)
                and np.isfinite(high_time)
                and high_time > low_time
            )
            interval_masks.append(int(observed))
            interval_values.append(float(high_time - low_time) if observed else 0.0)

        denominator = self.n_rows * self.x_sumsq - self.x_sum**2
        slope = (
            (self.n_rows * self.xv_sum - self.x_sum * self.value_sum[0]) / denominator
            if denominator > 0
            else 0.0
        )

        v80_observed = (
            self.soc_min <= 80.0 <= self.soc_max
            and np.isfinite(self.v80_lower_value)
            and np.isfinite(self.v80_upper_value)
        )
        if not v80_observed:
            v80 = 0.0
        elif self.v80_upper_soc == self.v80_lower_soc:
            v80 = self.v80_upper_value
        else:
            weight = (80.0 - self.v80_lower_soc) / (
                self.v80_upper_soc - self.v80_lower_soc
            )
            v80 = self.v80_lower_value + weight * (
                self.v80_upper_value - self.v80_lower_value
            )

        return {
            "car": self.car,
            "charge_segment": self.charge_segment,
            "capacity": self.capacity_max,
            "t40_50": interval_values[0],
            "t50_60": interval_values[1],
            "t60_70": interval_values[2],
            "t70_80": interval_values[3],
            "t80_90": interval_values[4],
            "t90_100": interval_values[5],
            "voltage_mean": means[0],
            "voltage_std": stds[0],
            "current_mean": means[1],
            "current_std": stds[1],
            "max_temp_mean": means[4],
            "min_temp_std": stds[5],
            "voltage_slope": slope,
            "max_cell_voltage_at_80": v80,
            "max_voltage_mean": means[2],
            "max_voltage_std": stds[2],
            "min_voltage_mean": means[3],
            "min_voltage_std": stds[3],
            "min_temp_mean": means[5],
            "voltage_diff_mean": means[6],
            "voltage_diff_std": stds[6],
            "temp_diff_std": stds[7],
            "mileage": self.mileage_max,
            "mask_t40_50": interval_masks[0],
            "mask_t50_60": interval_masks[1],
            "mask_t60_70": interval_masks[2],
            "mask_t70_80": interval_masks[3],
            "mask_t80_90": interval_masks[4],
            "mask_t90_100": interval_masks[5],
            "mask_v80": int(v80_observed),
            "n_rows": self.n_rows,
        }


def collect_cycles(args: argparse.Namespace) -> dict[tuple[int, int], CycleAccumulator]:
    cycles: dict[tuple[int, int], CycleAccumulator] = {}
    total_files = 0
    labelled_files = 0
    malformed_files = 0

    archive = args.input_dir / ARCHIVE_NAME
    if not archive.is_file():
        raise FileNotFoundError(archive)
    LOGGER.info(
        "Streaming manufacturer %d: %s",
        MANUFACTURER_ID,
        archive,
    )
    for file_id, signal, metadata in iter_archive_pickles(
        archive, args.max_files_per_archive
    ):
        total_files += 1
        if args.progress_every and total_files % args.progress_every == 0:
            LOGGER.info(
                "%s snippets read; %s labelled; %s cycles",
                f"{total_files:,}",
                f"{labelled_files:,}",
                f"{len(cycles):,}",
            )
        try:
            capacity = float(metadata["capacity"])
            if not np.isfinite(capacity) or capacity <= 0:
                continue
            car = int(metadata["car"])
            charge_segment = int(metadata["charge_segment"])
            mileage = float(metadata["mileage"])
        except (KeyError, TypeError, ValueError):
            malformed_files += 1
            continue

        labelled_files += 1
        key = (car, charge_segment)
        accumulator = cycles.get(key)
        if accumulator is None:
            accumulator = CycleAccumulator(car=car, charge_segment=charge_segment)
            cycles[key] = accumulator
        accumulator.update(file_id, signal, capacity, mileage)
    LOGGER.info(
        "Completed %s snippets",
        f"{total_files:,}",
    )

    LOGGER.info(
        "Collected %s labelled cycles from %s snippets (%d malformed)",
        f"{len(cycles):,}",
        f"{labelled_files:,}",
        malformed_files,
    )
    return cycles


def build_candidates(
    cycles: dict[tuple[int, int], CycleAccumulator],
    history_length: int,
    prediction_length: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_rows: list[dict[str, float | int]] = []
    quality_rows: list[dict[str, float | int | bool]] = []

    for accumulator in cycles.values():
        quality = accumulator.finalize_quality()
        quality_rows.append(
            {
                "car": accumulator.car,
                "charge_segment": accumulator.charge_segment,
                "n_rows": accumulator.n_rows,
                "n_snippets": accumulator.n_snippets,
                "soc_min": accumulator.soc_min,
                "soc_max": accumulator.soc_max,
                **quality,
            }
        )
        if quality["usable"]:
            feature_rows.append(accumulator.feature_row())

    features = pd.DataFrame(feature_rows)
    quality_df = pd.DataFrame(quality_rows)
    if features.empty:
        raise RuntimeError("No usable labelled cycles were found")

    mask_coverage = features[MASK_COLUMNS].mean(axis=1)
    feature_stats = (
        features.assign(mask_coverage=mask_coverage)
        .groupby("car", as_index=False)
        .agg(
            usable_cycles=("charge_segment", "size"),
            mask_coverage=("mask_coverage", "mean"),
            median_rows=("n_rows", "median"),
            min_rows=("n_rows", "min"),
        )
    )
    quality_stats = (
        quality_df.groupby("car", as_index=False)
        .agg(
            labelled_cycles=("charge_segment", "size"),
            mean_monotonic_fraction=("monotonic_fraction", "mean"),
            max_soc_drop=("max_soc_drop", "max"),
            finite_cycle_ratio=("usable", "mean"),
        )
    )
    report = feature_stats.merge(quality_stats, on="car", how="left")
    report["model_windows"] = (
        report["usable_cycles"] - history_length - prediction_length + 1
    ).clip(lower=0)
    report = report.sort_values(
        [
            "model_windows",
            "usable_cycles",
            "finite_cycle_ratio",
            "mean_monotonic_fraction",
            "mask_coverage",
            "median_rows",
            "min_rows",
            "car",
        ],
        ascending=[False, False, False, False, False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    report.insert(0, "rank", np.arange(1, len(report) + 1))
    return features, report


def add_soh_columns(
    features: pd.DataFrame,
    selected_cars: list[int],
    max_imf: int,
) -> pd.DataFrame:
    try:
        from PyEMD import EMD
    except ImportError as error:
        raise RuntimeError(
            "EMD-signal is required to calculate SOH_ref; "
            "install the project dependencies"
        ) from error

    selected = features.loc[features["car"].isin(selected_cars)].copy()
    frames: list[pd.DataFrame] = []
    for car, vehicle in selected.groupby("car", sort=False):
        vehicle = vehicle.sort_values(
            "charge_segment", kind="stable"
        ).reset_index(drop=True)
        nominal_capacity = float(vehicle["capacity"].quantile(0.99))
        vehicle["SOH_hist"] = vehicle["capacity"] / nominal_capacity
        emd = EMD()
        emd.emd(vehicle["SOH_hist"].to_numpy(dtype=np.float64), max_imf=max_imf)
        _, residual = emd.get_imfs_and_residue()
        if len(residual) != len(vehicle) or not np.isfinite(residual).all():
            raise RuntimeError(f"Invalid EMD residual for car {car}")
        vehicle["SOH_ref"] = residual
        frames.append(vehicle)
    return pd.concat(frames, ignore_index=True)


def validate_output(df: pd.DataFrame, selected_cars: list[int]) -> None:
    if list(df.columns) != OUTPUT_COLUMNS:
        raise AssertionError("Output column order differs from the agreed schema")
    if df.isna().any().any() or not np.isfinite(df.select_dtypes("number")).all().all():
        raise AssertionError("Output contains NaN or infinite values")
    if df.duplicated(["car", "charge_segment"]).any():
        raise AssertionError("Output contains duplicate (car, charge_segment) rows")
    if set(df["car"].unique()) != set(selected_cars):
        raise AssertionError("Output vehicle IDs differ from the selection")
    for mask in MASK_COLUMNS:
        if not set(df[mask].unique()).issubset({0, 1}):
            raise AssertionError(f"{mask} is not binary")


def write_csv(df: pd.DataFrame, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    args = parse_args()

    LOGGER.info(
        "Raw directory: %s",
        args.input_dir,
    )

    if args.top_n <= 0:
        raise ValueError("--top-n must be positive")
    if args.history_length <= 0:
        raise ValueError("--history-length must be positive")
    if args.prediction_length <= 0:
        raise ValueError("--prediction-length must be positive")
    if args.max_imf <= 0:
        raise ValueError("--max-imf must be positive")
    if (
        args.max_files_per_archive is not None
        and args.max_files_per_archive <= 0
    ):
        raise ValueError("--max-files-per-archive must be positive")
    if args.output.resolve() == args.report.resolve():
        raise ValueError("--output and --report must be different paths")

    for path in (args.output, args.report):
        if path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite {path}; pass --overwrite"
            )

    cycles = collect_cycles(args)
    features, report = build_candidates(
        cycles, args.history_length, args.prediction_length
    )
    eligible = report.loc[report["model_windows"] > 0]
    if len(eligible) < args.top_n:
        raise RuntimeError(
            f"Only {len(eligible)} vehicles have at least "
            f"{args.history_length + args.prediction_length} usable cycles; "
            f"cannot select {args.top_n}."
        )

    selected_cars = (
        eligible.head(args.top_n)["car"].astype(int).tolist()
    )
    report["selected"] = report["car"].isin(selected_cars)

    output = add_soh_columns(features, selected_cars, args.max_imf)
    output = output[OUTPUT_COLUMNS].sort_values(
        ["car", "charge_segment"], kind="stable"
    ).reset_index(drop=True)
    validate_output(output, selected_cars)

    write_csv(output, args.output, args.overwrite)
    write_csv(report, args.report, args.overwrite)
    LOGGER.info(
        "Selected cars: %s",
        selected_cars,
    )
    LOGGER.info(
        "Saved dataset: %s",
        args.output,
    )
    LOGGER.info(
        "Final shape: %s",
        output.shape,
    )
    LOGGER.info(
        "Saved selection audit: %s",
        args.report,
    )


if __name__ == "__main__":
    main()
