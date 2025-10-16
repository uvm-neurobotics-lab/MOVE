"""Shared backbone models used by MOVE fitness metrics."""
from __future__ import annotations

from torchvision.models import vgg16, VGG16_Weights

FEATURE_EXTRACTOR = vgg16(weights=VGG16_Weights.DEFAULT).features
FEATURE_EXTRACTOR.eval()
