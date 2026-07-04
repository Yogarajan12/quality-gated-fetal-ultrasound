"""
Configuration loader.

Reads configs/default.yaml into a flat `Config` object whose attribute names
match the ones the original notebook already uses (config.csv_path,
config.image_size, config.noise_levels, ...), so the notebook can switch from
its inline dataclass to this loader with a one-line change:

    from config import load_config
    config = load_config()            # instead of: config = Config()

New fields exposed by the YAML (val_ratio, test_ratio, seed, dataset_url,
prior_model, target_coverage, ...) are available as attributes too.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml  # add `pyyaml` to requirements.txt


@dataclass
class Config:
    """Flat config consumed by the pipeline. Populated from YAML by load_config()."""

    # ---- Paths ----
    data_dir: str = "data/fetal_planes"
    images_dir: str = "data/fetal_planes/Images"
    csv_path: str = "data/fetal_planes/FETAL_PLANES_DB_data.csv"
    output_dir: str = "outputs"
    model_dir: str = "models"
    dataset_url: str = (
        "https://zenodo.org/records/3904280/files/FETAL_PLANES_ZENODO.zip?download=1"
    )

    # ---- Data ----
    image_size: int = 256
    num_classes: int = 6
    class_names: List[str] = field(default_factory=lambda: [
        "Fetal abdomen", "Fetal brain", "Fetal femur",
        "Fetal thorax", "Maternal cervix", "Other",
    ])
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42

    # ---- Training ----
    batch_size: int = 16
    num_epochs: int = 15
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    grad_clip_norm: float = 1.0

    # ---- Diffusion ----
    ddpm_num_inference_steps: int = 20
    noise_levels: List[int] = field(default_factory=lambda: [400])
    num_uncertainty_samples: int = 2
    prior_model: str = "google/ddpm-celebahq-256"

    # ---- Gating ----
    target_coverage: float = 0.85
    confidence_threshold: float = 0.60
    uncertainty_threshold: float = 0.15

    # ---- Runtime ----
    device: str = "cuda"
    num_workers: int = 2

    def __post_init__(self):
        # Resolve device, falling back to CPU if CUDA is unavailable.
        if self.device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    self.device = "cpu"
            except ImportError:
                self.device = "cpu"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)


def _default_config_path() -> Path:
    """configs/default.yaml resolved relative to this file."""
    return Path(__file__).resolve().parent / "configs" / "default.yaml"


def load_config(path: Optional[str] = None) -> Config:
    """
    Load a Config from YAML, flattening the nested sections.

    Falls back to dataclass defaults for any key the YAML omits, so a partial
    config file still works.
    """
    cfg_path = Path(path) if path is not None else _default_config_path()
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {cfg_path}. "
            "Pass a path explicitly: load_config('configs/default.yaml')."
        )

    with open(cfg_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    # Flatten nested sections into one kwargs dict for the dataclass.
    flat: dict = {}
    for section in ("paths", "data", "train", "diffusion", "gating", "runtime"):
        section_values = raw.get(section, {})
        if section_values:
            flat.update(section_values)

    # Keep only keys the dataclass knows about; ignore extras gracefully.
    known = set(Config.__dataclass_fields__.keys())
    filtered = {k: v for k, v in flat.items() if k in known}
    unknown = set(flat) - known
    if unknown:
        print(f"[config] ignoring unrecognized keys: {sorted(unknown)}")

    return Config(**filtered)


if __name__ == "__main__":
    config = load_config()
    print(f"Loaded config: {config.num_classes} classes, "
          f"image size {config.image_size}, device {config.device}")
