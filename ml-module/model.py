from __future__ import annotations

import torch
from torch import nn
from torchvision import models

MODEL_NAMES = ["tiny-cnn", "mobilenetv3-small"]


class TinyConvNet(nn.Module):
    def __init__(self, classCount: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, classCount),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        features = self.features(tensor)
        return self.classifier(features)


def buildModel(
    modelName: str,
    classCount: int,
    pretrained: bool,
    dropout: float = 0.35,
) -> nn.Module:
    if modelName == "tiny-cnn":
        return TinyConvNet(classCount=classCount)

    if modelName == "mobilenetv3-small":
        weights = None
        if pretrained:
            try:
                weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            except Exception as error:  # noqa: BLE001
                print(f"[warning] failed to load pretrained weights metadata: {error}")
                weights = None

        try:
            model = models.mobilenet_v3_small(weights=weights)
        except Exception as error:  # noqa: BLE001
            print(f"[warning] failed to initialize pretrained backbone, fallback to random init: {error}")
            model = models.mobilenet_v3_small(weights=None)

        inFeatures = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(inFeatures, 512),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, classCount),
        )
        return model

    raise ValueError(f"Unsupported model name: {modelName}")


def setBackboneFrozen(model: nn.Module, modelName: str, frozen: bool) -> None:
    if modelName != "mobilenetv3-small":
        return
    featureLayers = getattr(model, "features", None)
    if featureLayers is None:
        return
    for parameter in featureLayers.parameters():
        parameter.requires_grad = not frozen
