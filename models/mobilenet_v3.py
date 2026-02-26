"""
MobileNetV3 Backbone with Attention Mechanisms
"""
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from .attention import ECALayer, CBAM


class MobileNetV3WithAttention(nn.Module):
    """MobileNetV3 with ECA and CBAM attention modules"""

    def __init__(self, num_classes=3, pretrained=True, use_eca=True, use_cbam=True):
        super(MobileNetV3WithAttention, self).__init__()

        # Load pretrained MobileNetV3
        if pretrained:
            weights = MobileNet_V3_Large_Weights.DEFAULT
            self.backbone = mobilenet_v3_large(weights=weights)
        else:
            self.backbone = mobilenet_v3_large(weights=None)

        # Get feature extractor (remove classifier)
        self.features = self.backbone.features

        # Add attention modules at bottleneck layers
        self.use_eca = use_eca
        self.use_cbam = use_cbam

        if use_eca:
            # Add ECA after specific layers
            self.eca1 = ECALayer(channels=40)  # After layer 6
            self.eca2 = ECALayer(channels=112)  # After layer 12
            self.eca3 = ECALayer(channels=960)  # After layer 16 (final features)

        if use_cbam:
            # Add CBAM after specific layers
            self.cbam1 = CBAM(channels=40)
            self.cbam2 = CBAM(channels=112)
            self.cbam3 = CBAM(channels=960)  # After layer 16 (final features)

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(960, 1280),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        # Feature extraction with attention
        for i, layer in enumerate(self.features):
            x = layer(x)

            # Apply attention at specific layers
            if self.use_eca or self.use_cbam:
                if i == 6:  # After bottleneck 1
                    if self.use_eca:
                        x = self.eca1(x)
                    if self.use_cbam:
                        x = self.cbam1(x)
                elif i == 12:  # After bottleneck 2
                    if self.use_eca:
                        x = self.eca2(x)
                    if self.use_cbam:
                        x = self.cbam2(x)
                elif i == 16:  # After bottleneck 3
                    if self.use_eca:
                        x = self.eca3(x)
                    if self.use_cbam:
                        x = self.cbam3(x)

        # Global pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x
