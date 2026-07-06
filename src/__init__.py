"""
Quality-gated fetal ultrasound classification.

A ResNet-18 classifier paired with diffusion reconstruction uncertainty and
softmax confidence for calibrated, coverage-controlled human-AI deferral.

Module map:
    data                  dataset download, splits, transforms, dataloaders
    classifier            ResNet-18 model and training loop
    diffusion_uncertainty diffusion reconstruction-variance estimator
    baselines             MC Dropout and augmentation-ensemble uncertainty
    gating                quality-gated classifier and threshold calibration
    evaluation            metrics, plots, ablations, OOD, deployment simulation
    utils                 reproducibility helpers
"""

__version__ = "0.1.0"
__author__ = "Yogarajan Sivakumar"
