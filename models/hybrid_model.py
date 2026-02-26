"""
Hybrid Model: MobileNetV3 + Attention + LSTM
"""
import torch
import torch.nn as nn
from .mobilenet_v3 import MobileNetV3WithAttention
from .lstm_branch import LSTMBranch


class HybridFatigueDetector(nn.Module):
    """
    Hybrid model combining:
    - MobileNetV3 with ECA and CBAM for spatial feature extraction
    - LSTM for temporal sequence analysis
    """

    def __init__(self, num_classes=3, pretrained=True, use_eca=True,
                 use_cbam=True, use_lstm=True, lstm_hidden_size=128):
        super(HybridFatigueDetector, self).__init__()

        self.use_lstm = use_lstm

        # Spatial feature extractor
        self.spatial_net = MobileNetV3WithAttention(
            num_classes=num_classes,
            pretrained=pretrained,
            use_eca=use_eca,
            use_cbam=use_cbam
        )

        # Temporal feature extractor
        if use_lstm:
            self.temporal_net = LSTMBranch(
                input_size=2,  # EAR and MAR
                hidden_size=lstm_hidden_size,
                num_layers=2,
                num_classes=num_classes
            )

            # Fusion layer
            self.fusion = nn.Sequential(
                nn.Linear(num_classes * 2, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(64, num_classes)
            )

    def forward(self, image, sequence=None):
        """
        Args:
            image: Input image tensor (batch, 3, H, W)
            sequence: Temporal sequence tensor (batch, seq_len, 2) - optional
        Returns:
            Output logits (batch, num_classes)
        """
        # Spatial features
        spatial_out = self.spatial_net(image)

        if self.use_lstm and sequence is not None:
            # Temporal features
            temporal_out = self.temporal_net(sequence)

            # Fusion
            combined = torch.cat([spatial_out, temporal_out], dim=1)
            out = self.fusion(combined)
        else:
            out = spatial_out

        return out