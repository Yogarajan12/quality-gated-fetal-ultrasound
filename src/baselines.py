"""Baseline uncertainty estimators for the ablation comparison.

Softmax confidence is computed inline in evaluation.py; the two class-based
baselines live here.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


class MCDropoutUncertainty:
    """Monte Carlo Dropout: variance of softmax across stochastic forward passes."""

    def __init__(self, model: nn.Module, num_samples: int = 10, dropout_rate: float = 0.3):
        self.model = model
        self.num_samples = num_samples
        self.dropout_rate = dropout_rate

    def enable_dropout(self):
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    @torch.no_grad()
    def predict_with_uncertainty(self, images: torch.Tensor, device: str) -> Dict:
        self.model.eval()
        self.enable_dropout()
        images = images.to(device)
        all_probs = []
        for _ in range(self.num_samples):
            all_probs.append(F.softmax(self.model(images), dim=1).cpu())
        all_probs = torch.stack(all_probs, dim=0)
        mean_probs = all_probs.mean(dim=0)
        return {
            "predictions": mean_probs.argmax(dim=1),
            "confidences": mean_probs.max(dim=1).values,
            "uncertainties": all_probs.var(dim=0).mean(dim=1).numpy(),
        }


class DeepEnsembleUncertainty:
    """Pseudo-ensemble: variance of softmax across light augmentations."""

    def __init__(self, model: nn.Module, num_augmentations: int = 5):
        self.model = model
        self.num_augmentations = num_augmentations
        self.augment = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(10),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        ])

    @torch.no_grad()
    def predict_with_uncertainty(self, images: torch.Tensor, device: str) -> Dict:
        self.model.eval()
        images = images.to(device)
        all_probs = [F.softmax(self.model(images), dim=1).cpu()]
        for _ in range(self.num_augmentations - 1):
            aug = torch.stack([self.augment(img.cpu()) for img in images]).to(device)
            all_probs.append(F.softmax(self.model(aug), dim=1).cpu())
        all_probs = torch.stack(all_probs, dim=0)
        mean_probs = all_probs.mean(dim=0)
        return {
            "predictions": mean_probs.argmax(dim=1),
            "confidences": mean_probs.max(dim=1).values,
            "uncertainties": all_probs.var(dim=0).mean(dim=1).numpy(),
        }
