"""Quality-gated classifier: combine confidence and diffusion uncertainty to defer."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import Config
from src.diffusion_uncertainty import DiffusionUncertaintyEstimator
from src.utils import denormalize


class QualityGatedClassifier:
    """Wrap a classifier with a deferral rule driven by confidence and uncertainty."""

    def __init__(
        self,
        classifier: nn.Module,
        uncertainty_estimator: DiffusionUncertaintyEstimator,
        config: Config,
    ):
        self.classifier = classifier
        self.uncertainty_estimator = uncertainty_estimator
        self.config = config
        self.device = config.device
        self.uncertainty_threshold = config.uncertainty_threshold
        self.confidence_threshold = config.confidence_threshold

    @torch.no_grad()
    def predict_with_uncertainty(
        self, images: torch.Tensor, images_01: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """Predict labels, confidences, diffusion uncertainty, and a deferral mask."""
        self.classifier.eval()
        images = images.to(self.device)

        logits = self.classifier(images)
        probs = F.softmax(logits, dim=1)
        confidences, predictions = probs.max(dim=1)

        if images_01 is None:
            images_01 = denormalize(images)
        unc = self.uncertainty_estimator.compute_reconstruction_uncertainty(images_01)
        uncertainties = unc["reconstruction_variance"]

        should_defer = (uncertainties > self.uncertainty_threshold) | (
            confidences.cpu() < self.confidence_threshold
        )
        return {
            "predictions": predictions.cpu(),
            "confidences": confidences.cpu(),
            "uncertainties": uncertainties,
            "uncertainty_maps": unc["uncertainty_maps"],
            "should_defer": should_defer,
            "logits": logits.cpu(),
        }

    def calibrate_thresholds_with_precomputed(
        self,
        val_loader: DataLoader,
        precomputed_uncertainties: np.ndarray,
        target_coverage: float = 0.85,
    ) -> Dict:
        """Grid-search confidence/uncertainty thresholds at a target coverage.

        Uses pre-computed diffusion uncertainties so only classifier confidences
        need a forward pass. Maximizes accuracy on accepted cases within a small
        band around the target coverage.
        """
        print("\n" + "=" * 60)
        print("CALIBRATING DEFERRAL THRESHOLDS")
        print("=" * 60)

        all_conf, all_correct, all_labels, all_preds = [], [], [], []
        self.classifier.eval()
        with torch.no_grad():
            for images, labels, _ in tqdm(val_loader, desc="Getting predictions"):
                images = images.to(self.device)
                probs = F.softmax(self.classifier(images), dim=1)
                confidences, predictions = probs.max(dim=1)
                all_conf.extend(confidences.cpu().numpy())
                all_correct.extend((predictions.cpu() == labels).numpy())
                all_labels.extend(labels.numpy())
                all_preds.extend(predictions.cpu().numpy())

        all_conf = np.array(all_conf)
        all_unc = precomputed_uncertainties
        all_correct = np.array(all_correct)

        n = min(len(all_conf), len(all_unc), len(all_correct))
        all_conf, all_unc, all_correct = all_conf[:n], all_unc[:n], all_correct[:n]

        best_acc = 0
        best = (self.confidence_threshold, self.uncertainty_threshold)
        for conf_t in np.linspace(0.3, 0.95, 20):
            for unc_t in np.linspace(
                np.percentile(all_unc, 50), np.percentile(all_unc, 99), 20
            ):
                defer = (all_unc > unc_t) | (all_conf < conf_t)
                coverage = 1 - defer.mean()
                if target_coverage - 0.05 <= coverage <= target_coverage + 0.05:
                    if (~defer).sum() > 0:
                        acc = all_correct[~defer].mean()
                        if acc > best_acc:
                            best_acc = acc
                            best = (conf_t, unc_t)

        self.confidence_threshold, self.uncertainty_threshold = best
        defer = (all_unc > self.uncertainty_threshold) | (
            all_conf < self.confidence_threshold
        )
        print("\nCalibrated Thresholds:")
        print(f"  Confidence threshold:  {self.confidence_threshold:.3f}")
        print(f"  Uncertainty threshold: {self.uncertainty_threshold:.4f}")
        print("\nExpected Performance:")
        print(f"  Coverage: {100 * (1 - defer.mean()):.1f}%")
        print(f"  Accuracy on non-deferred: {100 * all_correct[~defer].mean():.1f}%")
        print(f"  Overall accuracy: {100 * all_correct.mean():.1f}%")
        return {
            "confidences": all_conf,
            "uncertainties": all_unc,
            "correct": all_correct,
            "labels": all_labels,
            "predictions": all_preds,
        }
