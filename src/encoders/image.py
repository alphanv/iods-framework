"""ImageEncoder — Section 3.1. ViT or EfficientNet, pre-trainable on iNaturalist."""

import torch
import torch.nn as nn


class ImageEncoder(nn.Module):
    def __init__(self, output_dim: int = 512, backbone: str = "vit_small"):
        super().__init__()
        # Placeholder — in practice use timm.create_model(backbone, pretrained=True)
        self.backbone = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.proj = nn.Linear(3, output_dim)  # placeholder dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        return self.proj(features)
