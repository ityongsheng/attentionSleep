"""
LSTM Branch for Temporal Feature Analysis
"""
import torch
import torch.nn as nn


class LSTMBranch(nn.Module):
    """LSTM network for temporal sequence analysis"""

    def __init__(self, input_size=2, hidden_size=128, num_layers=2, num_classes=3):
        """
        Args:
            input_size: Number of input features (EAR, MAR)
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
        """
        super(LSTMBranch, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, sequence_length, input_size)
        Returns:
            Output tensor of shape (batch, num_classes)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the last hidden state
        out = h_n[-1]  # Shape: (batch, hidden_size)

        # Classification
        out = self.fc(out)

        return out