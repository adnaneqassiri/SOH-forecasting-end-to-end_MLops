"""Reusable inference service for the held-out Data 4 test fleet."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import torch

from src.data.datasets import (
    DATA4_FEATURES_COLS,
    HISTORY_LENGTH,
    MASK_FEATURES_COLS,
    PREDICTION_LENGTH,
)
from src.inference.predict import (
    apply_saved_scaler,
    create_model,
    default_artifact_paths,
    estimate_nominal_capacity,
    forecast_latest_window,
    load_config,
    prepare_latest_window,
    project_path,
    resolve_device,
)
from src.training.evaluate import load_checkpoint


class TestFleetPredictor:
    """Load the model once and forecast fixed test-vehicle snapshots."""

    def __init__(
        self,
        *,
        config: dict[str, Any] | None = None,
        device_name: str | None = None,
        data_path: Path | None = None,
        checkpoint_path: Path | None = None,
        scaler_path: Path | None = None,
    ) -> None:
        self.config = config or load_config()
        configured_device = device_name or self.config["training"]["device"]
        self.device = resolve_device(configured_device)
        self.feature_cols = list(DATA4_FEATURES_COLS)
        self.test_vehicle_ids = tuple(
            int(value)
            for value in self.config["training"]["splits"]["data4"]["test"]
        )
        if not self.test_vehicle_ids:
            raise ValueError("The configured Data 4 test split is empty")
        configured_snapshots = self.config.get("inference", {}).get(
            "test_snapshot_indices", {}
        )
        self.snapshot_indices = {
            int(vehicle_id): int(snapshot_index)
            for vehicle_id, snapshot_index in configured_snapshots.items()
        }
        missing_snapshots = sorted(
            set(self.test_vehicle_ids) - set(self.snapshot_indices)
        )
        extra_snapshots = sorted(
            set(self.snapshot_indices) - set(self.test_vehicle_ids)
        )
        if missing_snapshots or extra_snapshots:
            raise ValueError(
                "Inference snapshots must exactly match the Data 4 test split; "
                f"missing={missing_snapshots}, extra={extra_snapshots}"
            )

        default_checkpoint, default_scaler = default_artifact_paths(self.config)
        self.checkpoint_path = Path(checkpoint_path or default_checkpoint)
        self.scaler_path = Path(scaler_path or default_scaler)
        configured_data = (
            Path(self.config["processed"]["real_ev"]) / "real_ev.csv"
        )
        self.data_path = project_path(data_path or configured_data)

        self.vehicles = self._load_test_vehicles()
        self.nominal_capacities = {
            vehicle_id: estimate_nominal_capacity(vehicle["capacity"])
            for vehicle_id, vehicle in self.vehicles.items()
        }
        for vehicle_id, vehicle in self.vehicles.items():
            vehicle["SOH_hist"] = (
                vehicle["capacity"] / self.nominal_capacities[vehicle_id]
            )
            self._validate_snapshot(vehicle_id, vehicle)

        if not self.scaler_path.is_file():
            raise FileNotFoundError(f"Scaler not found: {self.scaler_path}")
        self.scaler = joblib.load(self.scaler_path)
        # Validate artifact compatibility once during service startup.
        first_vehicle = self.vehicles[self.test_vehicle_ids[0]]
        apply_saved_scaler(first_vehicle.tail(HISTORY_LENGTH), self.scaler)

        self.model = create_model(self.config, self.device)
        load_checkpoint(self.model, self.checkpoint_path, self.device)
        self.model.eval()
        self._model_lock = Lock()

    def _load_test_vehicles(self) -> dict[int, pd.DataFrame]:
        if not self.data_path.is_file():
            raise FileNotFoundError(
                f"Processed Data 4 CSV not found: {self.data_path}"
            )
        dataframe = pd.read_csv(self.data_path).rename(
            columns={"car": "vehicle_id"}
        )
        required = {
            "vehicle_id",
            "charge_segment",
            "capacity",
            *DATA4_FEATURES_COLS,
            *MASK_FEATURES_COLS,
            "SOH_ref",
        }
        missing = sorted(required.difference(dataframe.columns))
        if missing:
            raise ValueError(
                f"Processed Data 4 CSV is missing columns: {missing}"
            )
        numeric = dataframe[list(required)].select_dtypes(include="number")
        if numeric.shape[1] != len(required):
            raise ValueError("Processed Data 4 model columns must be numeric")
        if dataframe[list(required)].isna().any().any():
            raise ValueError("Processed Data 4 model columns contain missing values")

        dataframe["vehicle_id"] = dataframe["vehicle_id"].astype(int)
        dataframe["charge_segment"] = dataframe["charge_segment"].astype(int)
        available = set(dataframe["vehicle_id"].unique())
        missing_vehicles = sorted(set(self.test_vehicle_ids) - available)
        if missing_vehicles:
            raise ValueError(
                "Configured test vehicles are absent from Data 4: "
                f"{missing_vehicles}"
            )

        vehicles: dict[int, pd.DataFrame] = {}
        for vehicle_id in self.test_vehicle_ids:
            vehicle = dataframe.loc[
                dataframe["vehicle_id"].eq(vehicle_id)
            ].sort_values("charge_segment", kind="stable").reset_index(drop=True)
            if vehicle["charge_segment"].duplicated().any():
                raise ValueError(
                    f"Vehicle {vehicle_id} contains duplicate charge segments"
                )
            if len(vehicle) < HISTORY_LENGTH:
                raise ValueError(
                    f"Vehicle {vehicle_id} has {len(vehicle)} events; "
                    f"{HISTORY_LENGTH} are required"
                )
            vehicles[vehicle_id] = vehicle.copy()
        return vehicles

    def _validate_snapshot(
        self,
        vehicle_id: int,
        vehicle: pd.DataFrame,
    ) -> None:
        snapshot_index = self.snapshot_indices[vehicle_id]
        if snapshot_index < HISTORY_LENGTH - 1:
            raise ValueError(
                f"Vehicle {vehicle_id} snapshot {snapshot_index} does not "
                f"provide {HISTORY_LENGTH} historical events"
            )
        required_rows = snapshot_index + 1 + PREDICTION_LENGTH
        if len(vehicle) < required_rows:
            raise ValueError(
                f"Vehicle {vehicle_id} snapshot {snapshot_index} requires "
                f"{required_rows} rows, but only {len(vehicle)} are available"
            )

    def list_vehicles(self) -> list[dict[str, Any]]:
        """Return only vehicles from the configured held-out test split."""
        return [
            {
                "vehicle_id": vehicle_id,
                "display_name": f"Véhicule {vehicle_id}",
                "available_cycles": len(self.vehicles[vehicle_id]),
                "snapshot_index": self.snapshot_indices[vehicle_id],
                "snapshot_charge_segment": int(
                    self.vehicles[vehicle_id]["charge_segment"].iloc[
                        self.snapshot_indices[vehicle_id]
                    ]
                ),
                "first_charge_segment": int(
                    self.vehicles[vehicle_id]["charge_segment"].iloc[0]
                ),
                "last_charge_segment": int(
                    self.vehicles[vehicle_id]["charge_segment"].iloc[-1]
                ),
            }
            for vehicle_id in self.test_vehicle_ids
        ]

    def _normalize_vehicle_ids(
        self,
        vehicle_ids: Iterable[int] | None,
    ) -> tuple[int, ...]:
        if vehicle_ids is None:
            return self.test_vehicle_ids
        normalized = tuple(dict.fromkeys(int(value) for value in vehicle_ids))
        if not normalized:
            raise ValueError("Select at least one test vehicle")
        unknown = sorted(set(normalized) - set(self.test_vehicle_ids))
        if unknown:
            raise KeyError(tuple(unknown))
        return normalized

    @lru_cache(maxsize=32)
    def forecast(self, vehicle_id: int) -> dict[str, Any]:
        """Forecast ten events from the vehicle's configured snapshot."""
        vehicle_id = int(vehicle_id)
        if vehicle_id not in self.vehicles:
            raise KeyError(vehicle_id)
        vehicle = self.vehicles[vehicle_id]
        snapshot_index = self.snapshot_indices[vehicle_id]
        observed = vehicle.iloc[: snapshot_index + 1].copy()
        future = vehicle.iloc[
            snapshot_index + 1 : snapshot_index + 1 + PREDICTION_LENGTH
        ]
        image, sequence, average_soh, _ = prepare_latest_window(
            observed, self.scaler
        )
        # Match the original retrospective demo: SOH_ref is available only
        # through the chosen cutoff and supplies the forecast anchor.
        current_soh = float(observed["SOH_ref"].iloc[-1])
        with self._model_lock:
            forecast = forecast_latest_window(
                self.model,
                image,
                sequence,
                average_soh,
                self.device,
                last_observed_soh=current_soh,
            )

        nominal_capacity = self.nominal_capacities[vehicle_id]
        predicted_capacities = forecast * nominal_capacity
        history_soh = observed["SOH_hist"].to_numpy(dtype=np.float64)
        reference_soh = observed["SOH_ref"].to_numpy(dtype=np.float64)
        history_cycles = observed["charge_segment"].astype(int).tolist()
        forecast_cycles = future["charge_segment"].astype(int).tolist()
        return {
            "vehicle_id": vehicle_id,
            "display_name": f"Véhicule {vehicle_id}",
            "nominal_capacity": float(nominal_capacity),
            "snapshot_index": snapshot_index,
            "history_length": HISTORY_LENGTH,
            "display_history_length": len(observed),
            "prediction_length": PREDICTION_LENGTH,
            "history_cycles": history_cycles,
            "history_offsets": list(range(-len(observed) + 1, 1)),
            "historical_soh": history_soh.tolist(),
            "reference_soh": reference_soh.tolist(),
            "current_soh": current_soh,
            "current_capacity": float(observed["capacity"].iloc[-1]),
            "forecast_horizons": list(range(1, PREDICTION_LENGTH + 1)),
            "forecast_cycles": forecast_cycles,
            "forecast": forecast.tolist(),
            "forecast_capacities": predicted_capacities.tolist(),
            "predicted_soh_10": float(forecast[-1]),
            "expected_change": float(forecast[-1] - current_soh),
        }

    def predict_many(
        self,
        vehicle_ids: Iterable[int] | None = None,
    ) -> dict[str, Any]:
        """Forecast selected test vehicles, or all test vehicles when omitted."""
        selected = self._normalize_vehicle_ids(vehicle_ids)
        predictions = [self.forecast(vehicle_id) for vehicle_id in selected]
        current = np.asarray(
            [prediction["current_soh"] for prediction in predictions],
            dtype=np.float64,
        )
        return {
            "selected_vehicle_ids": list(selected),
            "total_vehicles": len(predictions),
            "fleet_average_soh": float(current.mean()),
            "count_below_090": int(np.count_nonzero(current < 0.90)),
            "lowest_current_soh": float(current.min()),
            "vehicles": predictions,
        }


__all__ = ["TestFleetPredictor"]
