"""
Prepares  FD004 engine histories for LSTM inference.

The preprocessing pipeline applies the fitted operating-condition
models, condition-wise sensor normalization, one-hot encoding, and
final-window extraction used during model training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PreparedEngineSequence:
    """
    Store one model-ready sequence and its identifying information.
    """

    values: np.ndarray
    engine_id: int
    ending_cycle: int
    ending_condition: int
    original_timestep_count: int
    padded_timestep_count: int


class FD004Preprocessor:
    """
    Transform one raw FD004 engine history into an LSTM sequence.
    """

    def __init__(
        self,
        artifacts: Mapping[str, Any],
    ) -> None:
        """
        Initialize from previously fitted preprocessing artifacts.
        """
        self.settings_scaler = artifacts[
            "condition_settings_scaler"
        ]

        self.condition_model = artifacts[
            "condition_kmeans"
        ]

        self.sensor_means = artifacts[
            "sensor_means"
        ]

        self.sensor_stds = artifacts[
            "sensor_stds"
        ]

        self.setting_columns = list(
            artifacts["setting_columns"]
        )

        self.sensor_columns = list(
            artifacts["sensor_columns"]
        )

        self.condition_columns = list(
            artifacts["condition_columns"]
        )

        self.sequence_feature_columns = list(
            artifacts[
                "sequence_feature_columns"
            ]
        )

        self.window = int(
            artifacts["window"]
        )

        self.rul_cap = float(
            artifacts["rul_cap"]
        )

    @classmethod
    def from_joblib(
        cls,
        path: str | Path,
    ) -> FD004Preprocessor:
        """
        Load fitted preprocessing artifacts from a joblib file.

        Alternative way of constructing object of class
        """
        artifacts = joblib.load(
            Path(path)
        )

        return cls(artifacts)

    def _validate_engine_history(
        self,
        engine_history: pd.DataFrame,
    ) -> None:
        """
        Validate required columns and ensure exactly one engine exists.
        """
        if engine_history.empty:
            raise ValueError(
                "Engine history cannot be empty."
            )

        required_columns = {
            "engine_id",
            "cycle",
            *self.setting_columns,
            *self.sensor_columns,
        }

        missing_columns = (
            required_columns
            - set(engine_history.columns)
        )

        if missing_columns:
            missing_text = ", ".join(
                sorted(missing_columns)
            )

            raise ValueError(
                "Engine history is missing required columns: "
                f"{missing_text}"
            )

        unique_engine_ids = (
            engine_history["engine_id"]
            .unique()
        )

        if len(unique_engine_ids) != 1:
            raise ValueError(
                "Inference requires rows from exactly one engine."
            )

        if engine_history["cycle"].duplicated().any():
            raise ValueError(
                "Engine history contains duplicate cycle values."
            )

        numeric_values = engine_history[
            self.setting_columns
            + self.sensor_columns
        ].to_numpy(
            dtype=np.float64
        )

        if not np.isfinite(
            numeric_values
        ).all():
            raise ValueError(
                "Engine history contains NaN or infinite values."
            )

    def transform_engine_history(
        self,
        engine_history: pd.DataFrame,
    ) -> PreparedEngineSequence:
        """
        Convert raw rows for one engine into one model-ready sequence.

        Processing steps:
            1. Validate and sort rows by cycle.
            2. Standardize operating settings.
            3. Predict each row's operating condition.
            4. Normalize sensors using that condition's statistics.
            5. Add one-hot condition features.
            6. take the final configured window.
            7. Left-pad short histories using their first observation.
        """
        self._validate_engine_history(
            engine_history
        )

        transformed = (
            engine_history
            .sort_values("cycle")
            .copy()
        )

        engine_id = int(
            transformed["engine_id"].iloc[0]
        )

        ending_cycle = int(
            transformed["cycle"].iloc[-1]
        )

        original_timestep_count = len(
            transformed
        )

        # The scaler was fitted only on FD004 training settings.
        settings_scaled = (
            self.settings_scaler.transform(
                transformed[
                    self.setting_columns
                ]
            )
        )

        transformed["operating_condition"] = (
            self.condition_model.predict(
                settings_scaled
            )
        ).astype(np.int64)

        row_conditions = transformed[
            "operating_condition"
        ].to_numpy()

        try:
            row_means = self.sensor_means.loc[
                row_conditions,
                self.sensor_columns,
            ].to_numpy()

            row_stds = self.sensor_stds.loc[
                row_conditions,
                self.sensor_columns,
            ].to_numpy()

        except KeyError as error:
            raise ValueError(
                "The condition model produced a condition without "
                "stored sensor statistics."
            ) from error

        if np.any(row_stds <= 0):
            raise ValueError(
                "Stored sensor standard deviations must be positive."
            )

        transformed[
            self.sensor_columns
        ] = (
            transformed[
                self.sensor_columns
            ].to_numpy()
            - row_means
        ) / row_stds

        for condition, column in enumerate(
            self.condition_columns
        ):
            transformed[column] = (
                transformed[
                    "operating_condition"
                ]
                == condition
            ).astype(np.float32)

        feature_values = transformed[
            self.sequence_feature_columns
        ].to_numpy(
            dtype=np.float32
        )

        if not np.isfinite(
            feature_values
        ).all():
            raise ValueError(
                "Preprocessing produced NaN or infinite values."
            )

        if len(feature_values) >= self.window:
            sequence = feature_values[
                -self.window:
            ]

            padded_timestep_count = 0

        else:
            padded_timestep_count = (
                self.window
                - len(feature_values)
            )

            left_padding = np.repeat(
                feature_values[[0]],
                repeats=padded_timestep_count,
                axis=0,
            )

            sequence = np.concatenate(
                [
                    left_padding,
                    feature_values,
                ],
                axis=0,
            )

        ending_condition = int(
            transformed[
                "operating_condition"
            ].iloc[-1]
        )

        # Add the batch dimension expected by the LSTM:
        # window × features -> 1 × window × features.
        sequence = sequence[
            np.newaxis,
            :,
            :,
        ]

        return PreparedEngineSequence(
            values=sequence,
            engine_id=engine_id,
            ending_cycle=ending_cycle,
            ending_condition=(
                ending_condition
            ),
            original_timestep_count=(
                original_timestep_count
            ),
            padded_timestep_count=(
                padded_timestep_count
            ),
        )