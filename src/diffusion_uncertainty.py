"""Diffusion reconstruction-uncertainty estimator.

Adds noise to an image and denoises it several times; the variance across
reconstructions is used as an "unusualness" score. NOTE: the default prior is
an out-of-domain face model (see configs/default.yaml) and, per the ablation in
the README, this signal underperforms simple baselines for fetal ultrasound.
Swapping in a fetal-US diffusion model is the key improvement.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMPipeline, UNet2DModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import Config
from src.utils import denormalize


class DiffusionUncertaintyEstimator:
    """Estimate per-image reconstruction uncertainty from a diffusion prior."""

    def __init__(self, config: Config):
        self.config = config
        self.device = config.device

        print("\n" + "=" * 60)
        print("LOADING DIFFUSION MODEL")
        print("=" * 60)
        try:
            self.pipeline = DDPMPipeline.from_pretrained(
                config.prior_model, torch_dtype=torch.float32
            ).to(self.device)
            print(f"Loaded pretrained DDPM pipeline: {config.prior_model}")
        except Exception as e:
            print(f"Could not load pretrained model: {e}")
            print("Creating custom UNet for diffusion...")
            self._create_custom_unet()

        self.scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            clip_sample=True,
        )
        self.scheduler.set_timesteps(config.ddpm_num_inference_steps)

    def _create_custom_unet(self):
        self.unet = UNet2DModel(
            sample_size=self.config.image_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(64, 128, 256, 256),
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        ).to(self.device)
        self.pipeline = None
        n = sum(p.numel() for p in self.unet.parameters())
        print(f"Created custom UNet with {n:,} parameters")

    @torch.no_grad()
    def add_noise(self, images: torch.Tensor, timestep: int):
        noise = torch.randn_like(images)
        alpha_prod = self.scheduler.alphas_cumprod[timestep]
        sqrt_alpha = alpha_prod ** 0.5
        sqrt_one_minus = (1 - alpha_prod) ** 0.5
        return sqrt_alpha * images + sqrt_one_minus * noise, noise

    @torch.no_grad()
    def denoise(self, noisy_images: torch.Tensor, start_timestep: int) -> torch.Tensor:
        images = noisy_images.clone()
        timesteps = self.scheduler.timesteps[self.scheduler.timesteps <= start_timestep]
        for t in timesteps:
            t_batch = torch.tensor([t] * images.shape[0], device=self.device)
            unet = self.pipeline.unet if self.pipeline is not None else self.unet
            noise_pred = unet(images, t_batch).sample
            images = self.scheduler.step(noise_pred, t, images).prev_sample
        return images.clamp(0, 1)

    @torch.no_grad()
    def compute_reconstruction_uncertainty(
        self,
        images: torch.Tensor,
        noise_levels: Optional[List[int]] = None,
        num_samples: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return reconstruction error, variance, and per-pixel uncertainty maps."""
        noise_levels = noise_levels or self.config.noise_levels
        num_samples = num_samples or self.config.num_uncertainty_samples

        images = images.to(self.device)
        all_reconstructions, all_errors = [], []

        for noise_level in noise_levels:
            level_recons = []
            for _ in range(num_samples):
                noisy, _ = self.add_noise(images, noise_level)
                reconstructed = self.denoise(noisy, noise_level)
                level_recons.append(reconstructed)
                error = F.mse_loss(reconstructed, images, reduction="none")
                all_errors.append(error.mean(dim=[1, 2, 3]))
            all_reconstructions.append(torch.stack(level_recons, dim=1))

        all_recons = torch.cat(all_reconstructions, dim=1)
        reconstruction_variance = all_recons.var(dim=1).mean(dim=[1, 2, 3])
        reconstruction_error = torch.stack(all_errors, dim=1).mean(dim=1)
        uncertainty_maps = all_recons.var(dim=1)

        return {
            "reconstruction_error": reconstruction_error.cpu(),
            "reconstruction_variance": reconstruction_variance.cpu(),
            "uncertainty_maps": uncertainty_maps.cpu(),
            "mean_reconstruction": all_recons.mean(dim=1).cpu(),
        }

    def compute_batch_uncertainty(
        self, dataloader: DataLoader, max_batches: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute uncertainty across a dataloader (inputs are denormalized to [0,1])."""
        all_variances, all_errors = [], []
        print("Computing diffusion-based uncertainty...")
        for i, (images, _, _) in enumerate(tqdm(dataloader)):
            if max_batches and i >= max_batches:
                break
            images_01 = denormalize(images)
            results = self.compute_reconstruction_uncertainty(images_01)
            all_variances.extend(results["reconstruction_variance"].numpy())
            all_errors.extend(results["reconstruction_error"].numpy())
        return np.array(all_variances), np.array(all_errors)


def load_or_compute_uncertainties(
    estimator: DiffusionUncertaintyEstimator,
    loader: DataLoader,
    save_path: str,
    name: str = "",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load cached uncertainties if present, else compute and cache them."""
    if os.path.exists(save_path):
        print(f"Loading pre-computed {name} uncertainties from {save_path}")
        return np.load(save_path), None
    print(f"Computing {name} uncertainties (this may take a while)...")
    uncertainties, errors = estimator.compute_batch_uncertainty(loader)
    np.save(save_path, uncertainties)
    print(f"Saved {name} uncertainties to {save_path}")
    return uncertainties, errors
