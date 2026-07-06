"""ResNet-18 baseline classifier and its training loop."""

from __future__ import annotations

import os
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import models
from tqdm.auto import tqdm

from config import Config


class FetalPlaneClassifier(nn.Module):
    """ResNet-18 with an ImageNet backbone and a small classification head."""

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.backbone = models.resnet18(
            weights="IMAGENET1K_V1" if pretrained else None
        )
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

    def get_features(self, x):
        """Return the pooled penultimate features (used for t-SNE)."""
        b = self.backbone
        x = b.conv1(x)
        x = b.bn1(x)
        x = b.relu(x)
        x = b.maxpool(x)
        x = b.layer1(x)
        x = b.layer2(x)
        x = b.layer3(x)
        x = b.layer4(x)
        x = b.avgpool(x)
        return torch.flatten(x, 1)


def train_classifier(
    model: nn.Module, train_loader, val_loader, config: Config
) -> Tuple[nn.Module, Dict[str, list]]:
    """Train with OneCycleLR, label smoothing, and gradient clipping.

    Saves the best-by-validation checkpoint to config.model_dir and reloads it
    before returning.
    """
    print("\n" + "=" * 60)
    print("TRAINING BASELINE CLASSIFIER")
    print("=" * 60)

    model = model.to(config.device)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.learning_rate * 10,
        epochs=config.num_epochs,
        steps_per_epoch=len(train_loader),
    )

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    ckpt_path = os.path.join(config.model_dir, "best_classifier.pth")

    for epoch in range(config.num_epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")
        for images, labels, _ in pbar:
            images, labels = images.to(config.device), labels.to(config.device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{100. * correct / total:.2f}%"}
            )

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(config.device), labels.to(config.device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total
        val_acc = 100. * val_correct / val_total
        history["train_loss"].append(train_loss / len(train_loader))
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss / len(val_loader))
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch + 1}: Train Acc {train_acc:.2f}%, Val Acc {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            print(f"  --> Saved new best model (Val Acc: {val_acc:.2f}%)")

    model.load_state_dict(torch.load(ckpt_path))
    print(f"\nTraining complete. Best Val Accuracy: {best_val_acc:.2f}%")
    return model, history
