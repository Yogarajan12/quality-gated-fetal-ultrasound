"""Dataset download, splitting, transforms, dataloaders, and EDA plots.

Paths come from the Config object (configs/default.yaml), so nothing is
hardcoded to a particular environment.
"""

from __future__ import annotations

import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from config import Config


def download_and_extract_dataset(config: Config) -> str:
    """Download and extract FETAL_PLANES_DB from Zenodo into config.data_dir.

    Returns the path to the extracted Images directory. Skips the download if
    the images already exist.
    """
    data_dir = config.data_dir
    images_dir = config.images_dir
    zip_path = os.path.join(data_dir, "FETAL_PLANES_ZENODO.zip")

    if os.path.exists(images_dir) and len(os.listdir(images_dir)) > 1000:
        print(f"Dataset already extracted: {len(os.listdir(images_dir))} images")
        return images_dir

    os.makedirs(data_dir, exist_ok=True)

    print("=" * 60)
    print("DOWNLOADING DATASET FROM ZENODO")
    print("=" * 60)
    print(f"URL: {config.dataset_url}")
    print("Size: ~2.5 GB | Estimated time: 3-5 minutes")

    def progress_hook(count, block_size, total_size):
        if count % 100 == 0:
            print(".", end="", flush=True)

    urllib.request.urlretrieve(config.dataset_url, zip_path, reporthook=progress_hook)
    print("\nDownload complete. Extracting...")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)
    print("Extraction complete.")

    # Handle possible nested folder structures in the archive.
    candidate_image_dirs = [
        images_dir,
        os.path.join(data_dir, "FETAL_PLANES_ZENODO", "Images"),
        os.path.join(data_dir, "FETAL_PLANES_DB", "Images"),
    ]
    for path in candidate_image_dirs:
        if os.path.exists(path) and len(os.listdir(path)) > 100:
            if path != images_dir:
                shutil.move(path, images_dir)
            break

    # Locate and copy the CSV to the configured path if needed.
    candidate_csvs = [
        os.path.join(data_dir, "FETAL_PLANES_DB_data.csv"),
        os.path.join(data_dir, "FETAL_PLANES_ZENODO", "FETAL_PLANES_DB_data.csv"),
        os.path.join(data_dir, "FETAL_PLANES_DB", "FETAL_PLANES_DB_data.csv"),
    ]
    if not os.path.exists(config.csv_path):
        for csv_path in candidate_csvs:
            if os.path.exists(csv_path):
                os.makedirs(os.path.dirname(config.csv_path), exist_ok=True)
                shutil.copy(csv_path, config.csv_path)
                print(f"Copied CSV to {config.csv_path}")
                break

    if os.path.exists(zip_path):
        os.remove(zip_path)

    n = len(os.listdir(images_dir)) if os.path.exists(images_dir) else 0
    print(f"Ready: {n} images in {images_dir}")
    return images_dir


class FetalPlanesDataset(Dataset):
    """PyTorch Dataset for FETAL_PLANES_DB with patient-stratified splits."""

    def __init__(
        self,
        csv_path: str,
        images_dir: str,
        transform=None,
        split: str = "train",
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ):
        self.images_dir = Path(images_dir)
        self.transform = transform

        # Zenodo ships a semicolon-separated CSV; fall back to comma.
        try:
            self.df = pd.read_csv(csv_path, sep=";")
            if len(self.df.columns) <= 1:
                self.df = pd.read_csv(csv_path, sep=",")
        except Exception:
            self.df = pd.read_csv(csv_path, sep=",")

        # Normalize column naming across dataset variants.
        if "Image_name" not in self.df.columns:
            for col in self.df.columns:
                if "image" in col.lower() or "file" in col.lower():
                    self.df["Image_name"] = self.df[col]
                    break
        if "Plane" not in self.df.columns:
            for col in self.df.columns:
                if "plane" in col.lower() and "brain" not in col.lower():
                    self.df["Plane"] = self.df[col]
                    break

        unique_planes = sorted(self.df["Plane"].unique())
        self.label_map = {plane: idx for idx, plane in enumerate(unique_planes)}
        self.idx_to_label = {idx: plane for plane, idx in self.label_map.items()}

        # Patient-stratified split when patient IDs are available (avoids leakage).
        if "Patient_num" in self.df.columns:
            patients = self.df["Patient_num"].unique()
            train_patients, temp_patients = train_test_split(
                patients, test_size=val_ratio + test_ratio, random_state=seed
            )
            val_patients, test_patients = train_test_split(
                temp_patients,
                test_size=test_ratio / (val_ratio + test_ratio),
                random_state=seed,
            )
            if split == "train":
                self.df = self.df[self.df["Patient_num"].isin(train_patients)]
            elif split == "val":
                self.df = self.df[self.df["Patient_num"].isin(val_patients)]
            else:
                self.df = self.df[self.df["Patient_num"].isin(test_patients)]
        else:
            train_df, temp_df = train_test_split(
                self.df,
                test_size=val_ratio + test_ratio,
                random_state=seed,
                stratify=self.df["Plane"],
            )
            val_df, test_df = train_test_split(
                temp_df,
                test_size=test_ratio / (val_ratio + test_ratio),
                random_state=seed,
                stratify=temp_df["Plane"],
            )
            self.df = {"train": train_df, "val": val_df}.get(split, test_df)

        self.df = self.df.reset_index(drop=True)
        print(f"{split.capitalize()} set: {len(self.df)} images")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["Image_name"]
        if not img_name.endswith(".png"):
            img_name = img_name + ".png"
        img_path = self.images_dir / img_name

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Warning: could not load {img_name}: {e}")
            image = Image.fromarray(
                np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            )

        if self.transform:
            image = self.transform(image)
        label = self.label_map[row["Plane"]]
        return image, label, img_name


def get_transforms(config: Config, is_training: bool = True):
    """Training-time augmentation vs deterministic eval transforms."""
    if is_training:
        return T.Compose([
            T.Resize((config.image_size, config.image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.3),
            T.RandomRotation(15),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return T.Compose([
        T.Resize((config.image_size, config.image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def create_dataloaders(
    config: Config,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int], Dict[int, str]]:
    """Build train/val/test dataloaders and update config with real class names."""
    print("\n" + "=" * 60)
    print("SETTING UP DATA")
    print("=" * 60)

    images_dir = download_and_extract_dataset(config)

    if not os.path.exists(config.csv_path):
        raise FileNotFoundError(f"CSV file not found at {config.csv_path}")

    common = dict(
        csv_path=config.csv_path,
        images_dir=images_dir,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
    )
    train_dataset = FetalPlanesDataset(
        transform=get_transforms(config, is_training=True), split="train", **common
    )
    val_dataset = FetalPlanesDataset(
        transform=get_transforms(config, is_training=False), split="val", **common
    )
    test_dataset = FetalPlanesDataset(
        transform=get_transforms(config, is_training=False), split="test", **common
    )

    config.num_classes = len(train_dataset.label_map)
    config.class_names = list(train_dataset.label_map.keys())

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True,
    )

    print("\nDataLoaders ready:")
    print(f"  Train: {len(train_loader)} batches ({len(train_dataset)} images)")
    print(f"  Val:   {len(val_loader)} batches ({len(val_dataset)} images)")
    print(f"  Test:  {len(test_loader)} batches ({len(test_dataset)} images)")

    return (
        train_loader, val_loader, test_loader,
        train_dataset.label_map, train_dataset.idx_to_label,
    )


def visualize_dataset(train_loader, val_loader, test_loader, label_map, config: Config):
    """Save a class-distribution plot for the three splits."""
    idx_to_label = {v: k for k, v in label_map.items()}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (loader, name) in zip(
        axes,
        [(train_loader, "Train"), (val_loader, "Validation"), (test_loader, "Test")],
    ):
        all_labels = []
        for _, labels, _ in loader:
            all_labels.extend(labels.numpy())
        unique, counts = np.unique(all_labels, return_counts=True)
        names = [idx_to_label[i][:12] for i in unique]
        ax.bar(names, counts, color="steelblue", edgecolor="black")
        ax.set_title(f"{name} Set (n={len(all_labels)})")
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    out = os.path.join(config.output_dir, "class_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
