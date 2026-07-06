"""Reproducibility and small shared helpers."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible runs.

    Note: diffusion sampling remains stochastic across Monte Carlo
    reconstructions, so uncertainty values still vary slightly between runs
    even with a fixed seed. Cached .npy uncertainties keep evaluation stable.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ImageNet normalization constants, reused across modules for denormalizing
# tensors back to [0, 1] before diffusion or visualization.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def denormalize(images: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization and clamp to [0, 1]. Expects [B, C, H, W]."""
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(images.device)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(images.device)
    return (images * std + mean).clamp(0, 1)
