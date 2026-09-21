"""
Define the LSTM architecture used for aircraft RUL prediction.

"""

from __future__ import annotations

import torch
from torch import nn


class RULPredictor(nn.Module):
    """
    Predicts one RUL value from each sensor sequence.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
    ) -> None:
        """
        Initializes the LSTM and final regression layer.

        Args:
            input_size:
                Number of features at each timestep.

            hidden_size:
                Number of values in the LSTM hidden representation.

            num_layers:
                Number of stacked LSTM layers.
        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        # batch_first=True tells PyTorch that tensor dimensions are ordered as:
        # (batch, timesteps, features)


        self.output_layer = nn.Linear(
            hidden_size,
            1,
        )

    def forward(
        self,
        sequences: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predicts one RUL value for every sequence in the batch.

        Args:
            sequences:
                Tensor shaped as batch × timesteps × features.

        Returns:
            Tensor containing one prediction per batch item.
        """
        lstm_output, _ = self.lstm(
            sequences
        )

        final_hidden = lstm_output[
            :,
            -1,
            :,
        ]

        predictions = self.output_layer(
            final_hidden
        )

        return predictions.squeeze(1)