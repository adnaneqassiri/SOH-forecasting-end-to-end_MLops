import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.data.datasets import (
    DATA4_FEATURES_COLS,
    HISTORY_LENGTH,
    MASKED_FEATURE_TO_MASK,
    MASK_FEATURES_COLS,
    MODEL_IMAGE_SHAPE,
)
from src.inference.predict import (
    estimate_nominal_capacity,
    extract_cycle_features,
    prepare_latest_window,
)


def make_raw_vehicle(number_of_cycles):
    rows = []
    soc_samples = [40, 50, 60, 70, 80, 90, 99]
    for charge_segment in range(number_of_cycles):
        capacity = 46.0 - 0.01 * charge_segment
        for sample, soc in enumerate(soc_samples):
            rows.append(
                {
                    "vehicle_id": "VIN-TEST",
                    "charge_block_id": charge_segment,
                    "capacity": capacity,
                    "mileage": 1_000.0 + charge_segment,
                    "volt": 3.5 + soc / 500.0,
                    "current": -18.0 + sample / 10.0,
                    "soc": soc,
                    "max_single_volt": 3.51 + soc / 500.0,
                    "min_single_volt": 3.49 + soc / 500.0,
                    "max_temp": 27.0 + sample / 10.0,
                    "min_temp": 24.0 + sample / 10.0,
                    "timestamp": sample * 600,
                }
            )
    return pd.DataFrame(rows)


def make_scaler(features):
    masked_features = list(MASKED_FEATURE_TO_MASK)
    ordinary_features = [
        feature
        for feature in DATA4_FEATURES_COLS
        if feature not in masked_features
    ]
    masked_stats = {}
    for feature, mask_column in MASKED_FEATURE_TO_MASK.items():
        observed = features.loc[features[mask_column].eq(1), feature]
        scale = float(observed.std(ddof=0))
        masked_stats[feature] = {
            "mask_column": mask_column,
            "mean": float(observed.mean()),
            "scale": scale if scale > 0 else 1.0,
        }
    return {
        "feature_cols": list(DATA4_FEATURES_COLS),
        "ordinary_features": ordinary_features,
        "ordinary_scaler": StandardScaler().fit(
            features[ordinary_features]
        ),
        "masked_feature_stats": masked_stats,
        "mask_columns": [],
    }


def test_raw_telemetry_is_aggregated_into_training_features():
    features, nominal_capacity = extract_cycle_features(
        make_raw_vehicle(2), nominal_capacity=46.0
    )

    assert nominal_capacity == 46.0
    assert features["charge_segment"].tolist() == [0, 1]
    assert features.loc[0, "SOH_hist"] == 1.0
    assert features.loc[0, "t40_50"] == 600.0
    assert features.loc[0, "t90_100"] == 600.0
    assert set(features[MASK_FEATURES_COLS].to_numpy().ravel()) == {1}


def test_nominal_capacity_matches_data4_processing_quantile():
    capacities = pd.Series([40.0, 42.0, 44.0, 46.0])

    assert estimate_nominal_capacity(capacities) == pytest.approx(
        capacities.quantile(0.99)
    )


def test_latest_window_uses_only_the_most_recent_history():
    features, _ = extract_cycle_features(
        make_raw_vehicle(HISTORY_LENGTH + 3), nominal_capacity=46.0
    )
    scaler = make_scaler(features)

    image, sequence, average_soh, history = prepare_latest_window(
        features, scaler
    )

    assert image.shape == (1, *MODEL_IMAGE_SHAPE)
    assert sequence.shape == (
        1,
        HISTORY_LENGTH,
        len(DATA4_FEATURES_COLS),
    )
    assert history["charge_segment"].tolist() == list(
        range(3, HISTORY_LENGTH + 3)
    )
    assert average_soh == pytest.approx(history["SOH_hist"].mean())
    assert np.isfinite(sequence.numpy()).all()


def test_latest_window_rejects_insufficient_history():
    features, _ = extract_cycle_features(
        make_raw_vehicle(HISTORY_LENGTH - 1), nominal_capacity=46.0
    )

    with pytest.raises(ValueError, match="At least 100 charging events"):
        prepare_latest_window(features, scaler={})
