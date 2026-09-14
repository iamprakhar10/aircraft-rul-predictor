"""
Test the shape and basic behaviour of the RUL LSTM.
"""

import torch

from aircraft_rul_predictor.model import (
    RULPredictor,
)


def test_rul_predictor_returns_one_value_per_sequence():
    """
    The model should return one prediction for every batch item.
    """
    model = RULPredictor(
        input_size=27,
        hidden_size=16,
        num_layers=1,
    )

    sequences = torch.randn(
        4,
        30,
        27,
    )

    predictions = model(
        sequences
    )

    assert predictions.shape == (4,)
    assert torch.isfinite(
        predictions
    ).all()