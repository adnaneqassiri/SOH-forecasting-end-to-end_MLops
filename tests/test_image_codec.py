import numpy as np
import pytest

from src.data.datasets import (
    HISTORY_LENGTH,
    MODEL_IMAGE_SHAPE,
    PREDICTION_LENGTH,
    image_to_soh,
    soh_to_image,
)


def test_curve_image_round_trip_stays_within_pixel_resolution():
    total_length = HISTORY_LENGTH + PREDICTION_LENGTH
    trajectory = np.linspace(0.97, 0.91, total_length, dtype=np.float32)
    average = float(trajectory[:HISTORY_LENGTH].mean())

    image = soh_to_image(trajectory, avg=average)
    decoded = image_to_soh(image, avg=average, n_points=total_length)

    assert image.shape == MODEL_IMAGE_SHAPE
    assert image.min() == 0.0
    assert image.max() == 1.0
    assert np.max(np.abs(decoded - trajectory)) < 0.001


def test_history_image_leaves_future_columns_blank():
    history = np.linspace(0.97, 0.93, HISTORY_LENGTH, dtype=np.float32)
    image = soh_to_image(history, avg=float(history.mean()))
    expected_last_column = round(
        (HISTORY_LENGTH - 1)
        * (MODEL_IMAGE_SHAPE[1] - 1)
        / (HISTORY_LENGTH + PREDICTION_LENGTH - 1)
    )

    assert np.any(image[:, expected_last_column] == 0.0)
    assert np.all(image[:, expected_last_column + 1 :] == 1.0)


def test_decoder_rejects_an_invalid_shape():
    with pytest.raises(ValueError, match="must have shape"):
        image_to_soh(np.ones((32, 32)), avg=0.95, n_points=110)
