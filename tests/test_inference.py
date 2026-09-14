"""
Test loading the saved-style pipeline and producing a prediction.
"""

import joblib
import numpy as np
import pandas as pd
import torch

from aircraft_rul_predictor.inference import (
    FD004InferenceService,
)
from aircraft_rul_predictor.model import (
    RULPredictor,
)


class IdentitySettingsScaler:
    """
    Return settings unchanged during the unit test.
    """

    def transform(self, values):
        return np.asarray(
            values,
            dtype=np.float64,
        )


class TwoConditionModel:
    """
    Assign one of two deterministic test conditions.
    """

    def predict(self, values):
        values = np.asarray(values)

        return (
            values[:, 0] > 0.5
        ).astype(np.int64)


def create_test_artifacts():
    """
    Build a small preprocessing bundle for inference testing.
    """
    sensor_columns = [
        f"sensor_{index}"
        for index in range(1, 22)
    ]

    condition_columns = [
        "condition_0",
        "condition_1",
    ]

    return {
        "condition_settings_scaler": (
            IdentitySettingsScaler()
        ),
        "condition_kmeans": (
            TwoConditionModel()
        ),
        "sensor_means": pd.DataFrame(
            0.0,
            index=[0, 1],
            columns=sensor_columns,
        ),
        "sensor_stds": pd.DataFrame(
            1.0,
            index=[0, 1],
            columns=sensor_columns,
        ),
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
    Create a valid raw history for one engine.
    """
    rows = []

    for cycle in range(1, 4):
        row = {
            "engine_id": 5,
            "cycle": cycle,
            "setting_1": float(
                cycle > 1
            ),
            "setting_2": 0.0,
            "setting_3": 0.0,
        }

        for sensor_index in range(1, 22):
            row[
                f"sensor_{sensor_index}"
            ] = float(cycle)

        rows.append(row)

    return pd.DataFrame(rows)


def test_inference_service_loads_and_predicts(
    tmp_path,
):
    """
    The service should load both files and return the known bias.
    """
    preprocessing_path = (
        tmp_path
        / "preprocessing.joblib"
    )

    model_path = (
        tmp_path
        / "model.pt"
    )

    artifacts = create_test_artifacts()

    joblib.dump(
        artifacts,
        preprocessing_path,
    )

    model = RULPredictor(
        input_size=23,
        hidden_size=4,
        num_layers=1,
    )

    # Zero every parameter and set the output bias. The model
    # must therefore predict exactly 12.5 for every input.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

        model.output_layer.bias.fill_(
            12.5
        )

    checkpoint = {
        "model_state_dict": (
            model.state_dict()
        ),
        "model_configuration": {
            "input_size": 23,
            "hidden_size": 4,
            "num_layers": 1,
        },
    }

    torch.save(
        checkpoint,
        model_path,
    )

    service = FD004InferenceService(
        model_path=model_path,
        preprocessing_path=(
            preprocessing_path
        ),
        device="cpu",
    )

    prediction = service.predict(
        create_engine_history()
    )

    assert prediction.engine_id == 5
    assert prediction.ending_cycle == 3
    assert prediction.ending_condition == 1
    assert prediction.raw_predicted_rul == 12.5
    assert prediction.predicted_rul == 12.5