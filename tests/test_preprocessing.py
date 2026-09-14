"""
Test condition assignment, normalization, one-hot encoding and padding.
"""

import numpy as np
import pandas as pd

from aircraft_rul_predictor.preprocessing import (
    FD004Preprocessor,
)


class IdentitySettingsScaler:
    """
    Return operating settings without changing them.
    """

    def transform(self, values):
        return np.asarray(
            values,
            dtype=np.float64,
        )


class TwoConditionModel:
    """
    Assign condition 1 when setting_1 is greater than 0.5.
    """

    def predict(self, values):
        values = np.asarray(values)

        return (
            values[:, 0] > 0.5
        ).astype(np.int64)


def create_test_artifacts():
    """
    Create deterministic preprocessing objects for unit tests.
    """
    sensor_columns = [
        f"sensor_{index}"
        for index in range(1, 22)
    ]

    condition_columns = [
        "condition_0",
        "condition_1",
    ]

    sensor_means = pd.DataFrame(
        0.0,
        index=[0, 1],
        columns=sensor_columns,
    )

    sensor_stds = pd.DataFrame(
        1.0,
        index=[0, 1],
        columns=sensor_columns,
    )

    return {
        "condition_settings_scaler": (
            IdentitySettingsScaler()
        ),
        "condition_kmeans": (
            TwoConditionModel()
        ),
        "sensor_means": sensor_means,
        "sensor_stds": sensor_stds,
        "setting_columns": [
            "setting_1",
            "setting_2",
            "setting_3",
        ],
        "sensor_columns": sensor_columns,
        "condition_columns": (
            condition_columns
        ),
        "sequence_feature_columns": (
            sensor_columns
            + condition_columns
        ),
        "window": 3,
        "rul_cap": 125,
    }


def create_engine_history():
    """
    Create two raw timesteps for one example engine.
    """
    rows = []

    for cycle, setting_1, sensor_value in [
        (1, 0.0, 1.0),
        (2, 1.0, 2.0),
    ]:
        row = {
            "engine_id": 10,
            "cycle": cycle,
            "setting_1": setting_1,
            "setting_2": 0.0,
            "setting_3": 0.0,
        }

        for sensor_index in range(1, 22):
            row[
                f"sensor_{sensor_index}"
            ] = sensor_value

        rows.append(row)

    return pd.DataFrame(rows)


def test_preprocessor_creates_padded_sequence():
    """
    A two-row history should be left-padded to a three-row window.
    """
    preprocessor = FD004Preprocessor(
        create_test_artifacts()
    )

    result = (
        preprocessor
        .transform_engine_history(
            create_engine_history()
        )
    )

    assert result.values.shape == (
        1,
        3,
        23,
    )

    assert result.engine_id == 10
    assert result.ending_cycle == 2
    assert result.ending_condition == 1
    assert result.original_timestep_count == 2
    assert result.padded_timestep_count == 1

    # The first observation is repeated for left padding.
    np.testing.assert_allclose(
        result.values[
            0,
            :,
            0,
        ],
        np.array(
            [1.0, 1.0, 2.0]
        ),
    )

    # One-hot conditions should be:
    # condition 0, condition 0, condition 1.
    np.testing.assert_array_equal(
        result.values[
            0,
            :,
            -2:,
        ],
        np.array([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]),
    )