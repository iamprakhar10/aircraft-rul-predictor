"""
Load the final FD004 pipeline and predict RUL for one engine.

This module combines saved preprocessing artifacts with the trained
PyTorch LSTM checkpoint. It can be imported from Python or invoked
through the project's command-line entry point.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aircraft_rul_predictor.model import (
    RULPredictor,
)
from aircraft_rul_predictor.preprocessing import (
    FD004Preprocessor,
)


@dataclass(frozen=True)
class RULPrediction:
    """
    Store the prediction and relevant inference metadata.
    """

    engine_id: int
    ending_cycle: int
    ending_condition: int
    timesteps_received: int
    padded_timesteps: int
    raw_predicted_rul: float
    predicted_rul: float
    rul_cap: float

    def to_dict(self) -> dict[str, int | float]:
        """
        Convert the result into a JSON-compatible dictionary.
        """
        return asdict(self)


def select_inference_device() -> torch.device:
    """
    Select CUDA, Apple MPS, or CPU in that priority order.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


class FD004InferenceService:
    """
    Run the complete saved FD004 RUL inference pipeline.
    """

    def __init__(
        self,
        model_path: str | Path,
        preprocessing_path: str | Path,
        device: str | torch.device | None = None,
    ) -> None:
        """
        Load the trained model and fitted preprocessing objects.
        """
        if device is None:
            selected_device = (
                select_inference_device()
            )
        else:
            selected_device = torch.device(
                device
            )

        self.device = selected_device

        self.preprocessor = (
            FD004Preprocessor.from_joblib(
                preprocessing_path
            )
        )

        checkpoint = torch.load(
            Path(model_path),
            map_location="cpu",
            weights_only=True,
        )

        model_configuration = checkpoint[
            "model_configuration"
        ]

        self.model = RULPredictor(
            input_size=int(
                model_configuration[
                    "input_size"
                ]
            ),
            hidden_size=int(
                model_configuration[
                    "hidden_size"
                ]
            ),
            num_layers=int(
                model_configuration[
                    "num_layers"
                ]
            ),
        )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        expected_input_size = len(
            self.preprocessor
            .sequence_feature_columns
        )

        if (
            model_configuration["input_size"]
            != expected_input_size
        ):
            raise ValueError(
                "Model input size does not match the saved "
                "preprocessing feature count."
            )

        self.model = self.model.to(
            self.device
        )

        self.model.eval()

    def predict(
        self,
        engine_history: pd.DataFrame,
    ) -> RULPrediction:
        """
        Predict capped RUL from raw chronological engine rows.
        """
        prepared = (
            self.preprocessor
            .transform_engine_history(
                engine_history
            )
        )

        sequence_tensor = torch.from_numpy(
            prepared.values
        ).to(
            self.device
        )

        with torch.no_grad():
            raw_prediction = float(
                self.model(
                    sequence_tensor
                )
                .cpu()
                .item()
            )

        # Provide an operationally valid displayed prediction.
        # The unmodified model output is retained separately.
        clipped_prediction = float(
            np.clip(
                raw_prediction,
                0.0,
                self.preprocessor.rul_cap,
            )
        )

        return RULPrediction(
            engine_id=prepared.engine_id,
            ending_cycle=(
                prepared.ending_cycle
            ),
            ending_condition=(
                prepared.ending_condition
            ),
            timesteps_received=(
                prepared.original_timestep_count
            ),
            padded_timesteps=(
                prepared.padded_timestep_count
            ),
            raw_predicted_rul=(
                raw_prediction
            ),
            predicted_rul=(
                clipped_prediction
            ),
            rul_cap=(
                self.preprocessor.rul_cap
            ),
        )


def main() -> None:
    """
    Run prediction from a CSV containing one raw engine history.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Predict capped RUL for one FD004 engine."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help=(
            "CSV containing raw rows for one engine."
        ),
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "models/fd004/final_safety_lstm.pt"
        ),
    )

    parser.add_argument(
        "--preprocessing",
        type=Path,
        default=Path(
            "models/fd004/preprocessing.joblib"
        ),
    )

    args = parser.parse_args()

    engine_history = pd.read_csv(
        args.input
    )

    service = FD004InferenceService(
        model_path=args.model,
        preprocessing_path=(
            args.preprocessing
        ),
    )

    prediction = service.predict(
        engine_history
    )

    print(
        json.dumps(
            prediction.to_dict(),
            indent=4,
        )
    )


if __name__ == "__main__":
    main()