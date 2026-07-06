"""End-to-end experiment runner.

Usage (from the repository root):
    python scripts/run_experiment.py --config configs/default.yaml

Pipeline: load data -> train ResNet-18 -> compute/load diffusion uncertainties
-> calibrate deferral thresholds -> evaluate -> plots -> extended analysis.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make the repository root importable so `import config` and `import src.*` work
# regardless of the current working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from config import load_config
from src.utils import set_seed
from src.data import create_dataloaders, visualize_dataset
from src.classifier import FetalPlaneClassifier, train_classifier
from src.diffusion_uncertainty import (
    DiffusionUncertaintyEstimator,
    load_or_compute_uncertainties,
)
from src.gating import QualityGatedClassifier
from src.evaluation import (
    evaluate_deferral_system_with_precomputed,
    plot_results,
    run_extended_analysis,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Quality-gated fetal US classification")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.seed)

    print("=" * 70)
    print("DIFFUSION-BASED RECONSTRUCTION UNCERTAINTY FOR FETAL US CLASSIFICATION")
    print("=" * 70)
    print(f"Device: {config.device} | Image size: {config.image_size}")

    train_loader, val_loader, test_loader, label_map, idx_to_label = create_dataloaders(config)
    visualize_dataset(train_loader, val_loader, test_loader, label_map, config)

    classifier = FetalPlaneClassifier(num_classes=config.num_classes)
    classifier, _ = train_classifier(classifier, train_loader, val_loader, config)

    estimator = DiffusionUncertaintyEstimator(config)
    val_unc_path = os.path.join(config.output_dir, "val_uncertainties.npy")
    test_unc_path = os.path.join(config.output_dir, "test_uncertainties.npy")
    val_uncertainties, _ = load_or_compute_uncertainties(
        estimator, val_loader, val_unc_path, "validation"
    )
    test_uncertainties, _ = load_or_compute_uncertainties(
        estimator, test_loader, test_unc_path, "test"
    )

    gated = QualityGatedClassifier(classifier, estimator, config)
    gated.calibrate_thresholds_with_precomputed(
        val_loader, val_uncertainties, target_coverage=config.target_coverage
    )

    results, metrics = evaluate_deferral_system_with_precomputed(
        gated, test_loader, test_uncertainties, idx_to_label, config
    )
    plot_results(results, metrics, config, gated)
    run_extended_analysis(
        classifier, gated, test_loader, results, test_uncertainties, idx_to_label, config
    )

    pd.DataFrame(
        {"metric": list(metrics.keys()), "value": list(metrics.values())}
    ).to_csv(os.path.join(config.output_dir, "final_metrics.csv"), index=False)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Overall Accuracy:      {100 * metrics['overall_accuracy']:.2f}%")
    print(f"Accuracy with Deferral:{100 * metrics['accuracy_on_non_deferred']:.2f}%")
    print(f"Improvement:           +{100 * (metrics['accuracy_on_non_deferred'] - metrics['overall_accuracy']):.2f}%")


if __name__ == "__main__":
    main()
