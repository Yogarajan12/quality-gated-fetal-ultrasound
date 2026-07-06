"""Evaluation, plotting, and the extended analyses (ablations, OOD, deployment).

All figures are written to config.output_dir. The functions mirror the notebook
but read cleaner and take pre-computed uncertainties where possible for speed.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.calibration import calibration_curve
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import Config
from src.baselines import DeepEnsembleUncertainty, MCDropoutUncertainty
from src.gating import QualityGatedClassifier
from src.utils import denormalize


# --------------------------------------------------------------------------- #
# Core deferral evaluation
# --------------------------------------------------------------------------- #
def evaluate_deferral_system_with_precomputed(
    gated: QualityGatedClassifier,
    test_loader: DataLoader,
    precomputed_uncertainties: np.ndarray,
    idx_to_label: Dict[int, str],
    config: Config,
) -> Tuple[Dict, Dict]:
    """Evaluate the gate on the test set using pre-computed uncertainties."""
    print("\n" + "=" * 60)
    print("EVALUATING DEFERRAL SYSTEM")
    print("=" * 60)

    conf, preds, labels = [], [], []
    gated.classifier.eval()
    with torch.no_grad():
        for images, y, _ in tqdm(test_loader, desc="Evaluating"):
            images = images.to(gated.device)
            probs = F.softmax(gated.classifier(images), dim=1)
            c, p = probs.max(dim=1)
            conf.extend(c.cpu().numpy())
            preds.extend(p.cpu().numpy())
            labels.extend(y.numpy())

    conf, preds, labels = np.array(conf), np.array(preds), np.array(labels)
    unc = precomputed_uncertainties
    n = min(len(conf), len(unc))
    conf, preds, labels, unc = conf[:n], preds[:n], labels[:n], unc[:n]

    correct = preds == labels
    defer = (unc > gated.uncertainty_threshold) | (conf < gated.confidence_threshold)

    results = {
        "predictions": preds, "labels": labels, "confidences": conf,
        "uncertainties": unc, "should_defer": defer, "correct": correct,
    }
    metrics = {
        "overall_accuracy": correct.mean(),
        "deferral_rate": defer.mean(),
        "coverage": 1 - defer.mean(),
        "accuracy_on_non_deferred": correct[~defer].mean() if (~defer).sum() else 0,
        "accuracy_on_deferred": correct[defer].mean() if defer.sum() else 0,
    }

    print("\n" + "-" * 40)
    print("FINAL RESULTS")
    print("-" * 40)
    print(f"Overall Accuracy:         {100 * metrics['overall_accuracy']:.2f}%")
    print(f"Deferral Rate:            {100 * metrics['deferral_rate']:.2f}%")
    print(f"Coverage:                 {100 * metrics['coverage']:.2f}%")
    print(f"Accuracy on Non-Deferred: {100 * metrics['accuracy_on_non_deferred']:.2f}%")
    print(f"Accuracy on Deferred:     {100 * metrics['accuracy_on_deferred']:.2f}%")
    print(f"--> Improvement from deferral: "
          f"{100 * (metrics['accuracy_on_non_deferred'] - metrics['overall_accuracy']):.2f}%")
    return results, metrics


def plot_results(results: Dict, metrics: Dict, config: Config, gated: QualityGatedClassifier):
    """Six-panel results summary figure."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    ax = axes[0, 0]
    thresholds = np.percentile(results["uncertainties"], np.linspace(0, 100, 50))
    cov, acc = [], []
    for t in thresholds:
        mask = results["uncertainties"] <= t
        if mask.sum() > 0:
            cov.append(mask.mean())
            acc.append(results["correct"][mask].mean())
    ax.plot(cov, acc, "b-", linewidth=2)
    ax.axhline(metrics["overall_accuracy"], color="r", linestyle="--", label="Baseline")
    ax.set_xlabel("Coverage"); ax.set_ylabel("Accuracy")
    ax.set_title("Coverage-Accuracy Curve"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    c = results["correct"].astype(bool)
    ax.hist(results["uncertainties"][c], bins=30, alpha=0.5, label="Correct", density=True)
    ax.hist(results["uncertainties"][~c], bins=30, alpha=0.5, label="Incorrect", density=True)
    ax.axvline(gated.uncertainty_threshold, color="r", linestyle="--", label="Threshold")
    ax.set_xlabel("Reconstruction Uncertainty"); ax.set_ylabel("Density")
    ax.set_title("Uncertainty Distribution"); ax.legend()

    ax = axes[0, 2]
    colors = ["green" if v else "red" for v in results["correct"]]
    ax.scatter(results["confidences"], results["uncertainties"], c=colors, alpha=0.3, s=10)
    ax.axhline(config.uncertainty_threshold, color="blue", linestyle="--")
    ax.axvline(config.confidence_threshold, color="blue", linestyle="--")
    ax.set_xlabel("Softmax Confidence"); ax.set_ylabel("Reconstruction Uncertainty")
    ax.set_title("Confidence vs Uncertainty")

    ax = axes[1, 0]
    mask = ~results["should_defer"]
    if mask.sum() > 0:
        cm = confusion_matrix(results["labels"][mask], results["predictions"][mask])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title("Confusion Matrix (Non-Deferred)")

    ax = axes[1, 1]
    prob_true, prob_pred = calibration_curve(
        results["correct"], results["confidences"], n_bins=10
    )
    ax.plot(prob_pred, prob_true, "o-", label="Model")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.set_xlabel("Mean Predicted Confidence"); ax.set_ylabel("Fraction Correct")
    ax.set_title("Calibration Curve"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 2]
    names = ["Overall\nAccuracy", "Non-Deferred\nAccuracy", "Coverage"]
    vals = [metrics["overall_accuracy"], metrics["accuracy_on_non_deferred"], metrics["coverage"]]
    bars = ax.bar(names, vals, color=["blue", "green", "orange"])
    ax.set_ylim(0, 1); ax.set_ylabel("Value"); ax.set_title("Summary Metrics")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{100 * v:.1f}%", ha="center")

    plt.tight_layout()
    out = os.path.join(config.output_dir, "results_summary.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


# --------------------------------------------------------------------------- #
# Extended analyses
# --------------------------------------------------------------------------- #
def compare_uncertainty_methods(classifier, gated, test_loader, test_uncertainties, config):
    """Compare diffusion uncertainty against MC Dropout, augmentation, and softmax."""
    print("\n" + "=" * 60)
    print("COMPARING UNCERTAINTY METHODS")
    print("=" * 60)
    device = config.device
    mc = MCDropoutUncertainty(classifier, num_samples=10)
    ens = DeepEnsembleUncertainty(classifier, num_augmentations=5)

    methods = {k: {"uncertainties": [], "correct": []} for k in
               ["Diffusion", "MC Dropout", "Augmentation Ensemble", "Softmax Confidence"]}
    idx = 0
    for images, labels, _ in tqdm(test_loader, desc="Comparing methods"):
        bs = images.shape[0]
        batch_diff = test_uncertainties[idx:idx + bs]; idx += bs

        mc_r = mc.predict_with_uncertainty(images, device)
        ens_r = ens.predict_with_uncertainty(images, device)

        classifier.eval()
        with torch.no_grad():
            probs = F.softmax(classifier(images.to(device)), dim=1)
            conf, preds = probs.max(dim=1)
            soft_unc = (1 - conf).cpu().numpy()
        correct = (preds.cpu() == labels).numpy()

        methods["Diffusion"]["uncertainties"].extend(batch_diff)
        methods["Diffusion"]["correct"].extend(correct)
        methods["MC Dropout"]["uncertainties"].extend(mc_r["uncertainties"])
        methods["MC Dropout"]["correct"].extend((mc_r["predictions"] == labels).numpy())
        methods["Augmentation Ensemble"]["uncertainties"].extend(ens_r["uncertainties"])
        methods["Augmentation Ensemble"]["correct"].extend((ens_r["predictions"] == labels).numpy())
        methods["Softmax Confidence"]["uncertainties"].extend(soft_unc)
        methods["Softmax Confidence"]["correct"].extend(correct)

    for m in methods:
        methods[m]["uncertainties"] = np.array(methods[m]["uncertainties"])
        methods[m]["correct"] = np.array(methods[m]["correct"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ["blue", "red", "green", "orange"]

    ax = axes[0, 0]
    for (name, data), color in zip(methods.items(), colors):
        u, c = data["uncertainties"], data["correct"]
        ts = np.percentile(u, np.linspace(0, 100, 50))
        cov, acc = [], []
        for t in ts:
            mask = u <= t
            if mask.sum() > 0:
                cov.append(mask.mean()); acc.append(c[mask].mean())
        ax.plot(cov, acc, "-", color=color, linewidth=2, label=name)
    ax.set_xlabel("Coverage"); ax.set_ylabel("Accuracy")
    ax.set_title("Coverage-Accuracy Curves by Method"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    aucs = []
    for name, data in methods.items():
        errs = (~data["correct"]).astype(int)
        fpr, tpr, _ = roc_curve(errs, data["uncertainties"])
        aucs.append(auc(fpr, tpr))
    bars = ax.bar(list(methods.keys()), aucs, color=colors, alpha=0.7, edgecolor="black")
    ax.set_ylabel("AUC-ROC"); ax.set_title("Error Detection AUC (higher is better)")
    ax.set_ylim(0.0, 1.0); ax.axhline(0.5, color="gray", linestyle="--")
    for bar, a in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{a:.3f}", ha="center", va="bottom")
    ax.set_xticklabels(list(methods.keys()), rotation=30, ha="right", fontsize=9)

    axes[0, 1].axis("off"); axes[1, 1].axis("off")
    plt.tight_layout()
    out = os.path.join(config.output_dir, "uncertainty_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")
    return methods


def per_class_analysis(results: Dict, idx_to_label: Dict[int, str], config: Config):
    """Per-class accuracy, deferral rate, and improvement from deferral."""
    print("\n" + "=" * 60)
    print("PER-CLASS DEFERRAL ANALYSIS")
    print("=" * 60)
    labels = results["labels"]; correct = results["correct"]
    defer = results["should_defer"]

    rows = {}
    for ci, cname in idx_to_label.items():
        mask = labels == ci
        if mask.sum() == 0:
            continue
        nd = ~defer[mask]
        rows[cname] = {
            "total": int(mask.sum()),
            "overall_acc": correct[mask].mean(),
            "deferral_rate": defer[mask].mean(),
            "acc_non_deferred": correct[mask][nd].mean() if nd.sum() else 0,
            "improvement": (correct[mask][nd].mean() - correct[mask].mean()) if nd.sum() else 0,
        }

    print(f"\n{'Class':<18}{'Total':<8}{'Acc':<9}{'NonDef':<9}{'Defer%':<9}{'Improv':<8}")
    print("-" * 60)
    for cname, m in rows.items():
        print(f"{cname:<18}{m['total']:<8}{100 * m['overall_acc']:<8.1f}%"
              f"{100 * m['acc_non_deferred']:<8.1f}%{100 * m['deferral_rate']:<8.1f}%"
              f"{100 * m['improvement']:+.1f}%")
    return rows


def run_extended_analysis(classifier, gated, test_loader, results, test_uncertainties,
                          idx_to_label, config):
    """Run the subset of extended analyses that are quick and robust."""
    print("\n" + "=" * 70)
    print("RUNNING EXTENDED ANALYSIS")
    print("=" * 70)
    methods = compare_uncertainty_methods(
        classifier, gated, test_loader, test_uncertainties, config
    )
    class_metrics = per_class_analysis(results, idx_to_label, config)
    clinical = clinical_metrics_analysis(results, idx_to_label, config)
    visualize_feature_space(classifier, test_loader, test_uncertainties, idx_to_label, config)
    ablation = ablation_study(gated.uncertainty_estimator, test_loader, config)
    ood_samples = create_ood_samples(test_loader, config)
    ood_results, ood_aucs = evaluate_ood_detection(
        gated.uncertainty_estimator, ood_samples, config
    )
    deployment = simulate_clinical_deployment(results, idx_to_label, config)
    print("\nExtended analysis complete.")
    return {
        "methods_comparison": methods,
        "class_metrics": class_metrics,
        "clinical_results": clinical,
        "ablation_results": ablation,
        "ood_results": ood_results,
        "ood_aucs": ood_aucs,
        "deployment_results": deployment,
    }


# --------------------------------------------------------------------------- #
# Clinical metrics (sensitivity / specificity across coverage)
# --------------------------------------------------------------------------- #
def clinical_metrics_analysis(results: Dict, idx_to_label: Dict[int, str], config: Config):
    """Per-class sensitivity/specificity at several coverage levels."""
    print("\n" + "=" * 60)
    print("CLINICAL METRICS ANALYSIS")
    print("=" * 60)
    labels = results["labels"]; preds = results["predictions"]; unc = results["uncertainties"]
    num_classes = len(idx_to_label)
    coverage_levels = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75]

    out = []
    for coverage in coverage_levels:
        if coverage < 1.0:
            thresh = np.percentile(unc, coverage * 100)
            mask = unc <= thresh
        else:
            mask = np.ones(len(labels), dtype=bool)

        per_class = {}
        for ci in range(num_classes):
            y_true = (labels[mask] == ci).astype(int)
            y_pred = (preds[mask] == ci).astype(int)
            tp = ((y_true == 1) & (y_pred == 1)).sum()
            tn = ((y_true == 0) & (y_pred == 0)).sum()
            fp = ((y_true == 0) & (y_pred == 1)).sum()
            fn = ((y_true == 1) & (y_pred == 0)).sum()
            per_class[idx_to_label[ci]] = {
                "sensitivity": tp / (tp + fn) if (tp + fn) else 0,
                "specificity": tn / (tn + fp) if (tn + fp) else 0,
            }
        out.append({
            "actual_coverage": mask.mean(),
            "accuracy": (preds[mask] == labels[mask]).mean(),
            "class_metrics": per_class,
        })

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    mean_sens, mean_spec, covs = [], [], []
    for r in out:
        mean_sens.append(np.mean([m["sensitivity"] for m in r["class_metrics"].values()]))
        mean_spec.append(np.mean([m["specificity"] for m in r["class_metrics"].values()]))
        covs.append(r["actual_coverage"])
    sc = ax.scatter(mean_spec, mean_sens, c=covs, cmap="viridis", s=100, edgecolors="black")
    plt.colorbar(sc, ax=ax, label="Coverage")
    ax.set_xlabel("Mean Specificity"); ax.set_ylabel("Mean Sensitivity")
    ax.set_title("Sensitivity vs Specificity Trade-off"); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(config.output_dir, "clinical_metrics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {path}")

    print(f"\n{'Coverage':<12}{'Accuracy':<12}{'Mean Sens':<12}{'Mean Spec':<12}")
    print("-" * 48)
    for r, s, sp in zip(out, mean_sens, mean_spec):
        print(f"{100 * r['actual_coverage']:<11.1f}%{100 * r['accuracy']:<11.1f}%"
              f"{100 * s:<11.1f}%{100 * sp:<11.1f}%")
    return out


# --------------------------------------------------------------------------- #
# Feature-space visualization (t-SNE)
# --------------------------------------------------------------------------- #
def visualize_feature_space(classifier, test_loader, test_uncertainties, idx_to_label,
                            config, max_samples: int = 1000):
    """t-SNE of penultimate features, colored by class, uncertainty, and correctness."""
    from sklearn.manifold import TSNE

    print("\n" + "=" * 60)
    print("FEATURE SPACE VISUALIZATION (t-SNE)")
    print("=" * 60)
    classifier.eval()
    feats, labels, correct = [], [], []
    count = 0
    with torch.no_grad():
        for images, y, _ in tqdm(test_loader, desc="Extracting features"):
            if count >= max_samples:
                break
            images = images.to(config.device)
            f = classifier.get_features(images)
            preds = classifier(images).argmax(dim=1)
            feats.append(f.cpu().numpy())
            labels.extend(y.numpy())
            correct.extend((preds.cpu() == y).numpy())
            count += len(y)

    feats = np.concatenate(feats, axis=0)[:max_samples]
    labels = np.array(labels)[:max_samples]
    correct = np.array(correct)[:max_samples]
    unc = test_uncertainties[:max_samples]

    print(f"Running t-SNE on {len(feats)} samples...")
    feats_2d = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(feats)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(idx_to_label)))
    for ci in range(len(idx_to_label)):
        m = labels == ci
        axes[0].scatter(feats_2d[m, 0], feats_2d[m, 1], c=[colors[ci]],
                        label=idx_to_label[ci][:12], alpha=0.6, s=20)
    axes[0].set_title("Feature Space by Class"); axes[0].legend(fontsize=8)
    axes[0].set_xticks([]); axes[0].set_yticks([])

    sc = axes[1].scatter(feats_2d[:, 0], feats_2d[:, 1], c=unc, cmap="Reds", alpha=0.6, s=20)
    plt.colorbar(sc, ax=axes[1], label="Uncertainty")
    axes[1].set_title("Feature Space by Uncertainty")
    axes[1].set_xticks([]); axes[1].set_yticks([])

    cc = ["green" if v else "red" for v in correct]
    axes[2].scatter(feats_2d[:, 0], feats_2d[:, 1], c=cc, alpha=0.6, s=20)
    axes[2].set_title("Feature Space: Correct vs Incorrect")
    axes[2].set_xticks([]); axes[2].set_yticks([])

    plt.tight_layout()
    path = os.path.join(config.output_dir, "feature_space_tsne.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {path}")


# --------------------------------------------------------------------------- #
# Ablation over diffusion parameters
# --------------------------------------------------------------------------- #
def ablation_study(estimator, test_loader, config, max_batches: int = 10):
    """Sweep noise levels and MC-sample counts; report mean/std uncertainty."""
    print("\n" + "=" * 60)
    print("ABLATION STUDY: DIFFUSION PARAMETERS")
    print("=" * 60)
    imgs = []
    for i, (images, _, _) in enumerate(test_loader):
        if i >= max_batches:
            break
        imgs.append(denormalize(images))
    imgs = torch.cat(imgs, dim=0)
    print(f"Running ablation on {len(imgs)} images...")

    configs = [
        {"noise_levels": [200], "num_samples": 3, "name": "1 level, 3 samples"},
        {"noise_levels": [400], "num_samples": 3, "name": "1 level (mid), 3 samples"},
        {"noise_levels": [200, 400], "num_samples": 3, "name": "2 levels, 3 samples"},
        {"noise_levels": [200, 400, 600], "num_samples": 3, "name": "3 levels, 3 samples"},
        {"noise_levels": [200, 400, 600], "num_samples": 5, "name": "3 levels, 5 samples"},
    ]
    results = []
    for cfg in tqdm(configs, desc="Ablation configs"):
        u = []
        for i in range(0, len(imgs), 8):
            batch = imgs[i:i + 8]
            r = estimator.compute_reconstruction_uncertainty(
                batch, noise_levels=cfg["noise_levels"], num_samples=cfg["num_samples"]
            )
            u.extend(r["reconstruction_variance"].numpy())
        u = np.array(u)
        results.append({"name": cfg["name"], "mean_unc": u.mean(), "std_unc": u.std()})

    fig, ax = plt.subplots(figsize=(9, 5))
    names = [r["name"] for r in results]
    means = [r["mean_unc"] for r in results]
    stds = [r["std_unc"] for r in results]
    ax.bar(range(len(names)), means, yerr=stds, capsize=5, color="steelblue",
           alpha=0.7, edgecolor="black")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Mean Uncertainty"); ax.set_title("Uncertainty by Configuration")
    plt.tight_layout()
    path = os.path.join(config.output_dir, "ablation_study.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {path}")

    print(f"\n{'Configuration':<28}{'Mean Unc':<12}{'Std Unc':<12}")
    print("-" * 52)
    for r in results:
        print(f"{r['name']:<28}{r['mean_unc']:<12.4f}{r['std_unc']:<12.4f}")
    return results


# --------------------------------------------------------------------------- #
# Out-of-distribution detection
# --------------------------------------------------------------------------- #
def create_ood_samples(test_loader, config):
    """Build corrupted variants of a normal batch to probe OOD sensitivity."""
    from torchvision.transforms.functional import gaussian_blur

    images, _, _ = next(iter(test_loader))
    images_01 = denormalize(images)
    ood = {"normal": images_01[:8]}
    for noise in [0.1, 0.2, 0.3]:
        ood[f"gaussian_noise_{noise}"] = (images_01 + torch.randn_like(images_01) * noise).clamp(0, 1)[:8]
    for k in [5, 11, 21]:
        ood[f"blur_k{k}"] = gaussian_blur(images_01, kernel_size=k)[:8]
    for factor in [0.3, 0.5]:
        ood[f"low_contrast_{factor}"] = (images_01 * factor + 0.5 * (1 - factor))[:8]
    ood["random"] = torch.rand_like(images_01[:8])
    ood["inverted"] = (1 - images_01[:8])
    return ood


def evaluate_ood_detection(estimator, ood_samples: Dict[str, torch.Tensor], config: Config):
    """Detection AUC of diffusion uncertainty for each corruption type vs normal."""
    print("\n" + "=" * 60)
    print("OUT-OF-DISTRIBUTION DETECTION ANALYSIS")
    print("=" * 60)
    results = {}
    for name, images in tqdm(ood_samples.items(), desc="Evaluating OOD"):
        u = estimator.compute_reconstruction_uncertainty(images)["reconstruction_variance"].numpy()
        results[name] = {"mean_uncertainty": u.mean(), "uncertainties": u}

    normal = results["normal"]["uncertainties"]
    aucs = {}
    for name, data in results.items():
        if name == "normal":
            continue
        y_true = np.array([0] * len(normal) + [1] * len(data["uncertainties"]))
        y_score = np.concatenate([normal, data["uncertainties"]])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        aucs[name] = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(9, 5))
    names = list(aucs.keys()); vals = list(aucs.values())
    ax.bar(range(len(names)), vals, color="salmon", edgecolor="black")
    ax.axhline(0.5, color="gray", linestyle="--", label="Random")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n[:12] for n in names], rotation=45, ha="right")
    ax.set_ylabel("Detection AUC"); ax.set_ylim(0, 1.05)
    ax.set_title("OOD Detection Performance (AUC)"); ax.legend()
    plt.tight_layout()
    path = os.path.join(config.output_dir, "ood_detection.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {path}")

    print(f"\n{'Corruption':<22}{'Mean Unc':<12}{'Detection AUC':<12}")
    print("-" * 46)
    print(f"{'normal':<22}{results['normal']['mean_uncertainty']:<12.4f}{'---':<12}")
    for name in aucs:
        print(f"{name:<22}{results[name]['mean_uncertainty']:<12.4f}{aucs[name]:<12.3f}")
    return results, aucs


# --------------------------------------------------------------------------- #
# Clinical deployment simulation
# --------------------------------------------------------------------------- #
def simulate_clinical_deployment(results: Dict, idx_to_label: Dict[int, str], config: Config,
                                 expert_accuracy: float = 0.95,
                                 expert_time_per_image: float = 30.0,
                                 model_time_per_image: float = 0.5):
    """Compare expert-only, model-only, and human-AI collaboration workflows."""
    print("\n" + "=" * 60)
    print("CLINICAL DEPLOYMENT SIMULATION")
    print("=" * 60)
    n = len(results["labels"])
    defer = results["should_defer"]
    model_correct = results["correct"]
    n_deferred = int(defer.sum()); n_auto = int((~defer).sum())

    expert_only_time = n * expert_time_per_image
    auto_correct = model_correct[~defer].sum() if n_auto else 0
    collab_correct = auto_correct + n_deferred * expert_accuracy
    collab_acc = collab_correct / n
    collab_time = n_auto * model_time_per_image + n_deferred * expert_time_per_image

    time_saved_pct = 100 * (expert_only_time - collab_time) / expert_only_time
    workload_reduction = 100 * (1 - n_deferred / n)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    scenarios = ["Expert Only", "Model Only", "Human-AI"]
    accs = [expert_accuracy, model_correct.mean(), collab_acc]
    axes[0].bar(scenarios, accs, color=["gold", "steelblue", "green"], edgecolor="black")
    axes[0].set_ylim(0, 1.1); axes[0].set_ylabel("Accuracy"); axes[0].set_title("Accuracy by Workflow")
    for i, a in enumerate(accs):
        axes[0].text(i, a + 0.02, f"{100 * a:.1f}%", ha="center", fontweight="bold")

    sizes = [n_auto, n_deferred]
    axes[1].pie(sizes, labels=[f"Auto ({n_auto})", f"Expert ({n_deferred})"],
                colors=["lightgreen", "lightyellow"], autopct="%1.1f%%", startangle=90)
    axes[1].set_title("Workload Distribution")
    plt.tight_layout()
    path = os.path.join(config.output_dir, "clinical_deployment.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {path}")

    print(f"\nExpert workload reduction: {workload_reduction:.1f}%")
    print(f"Time saved: {time_saved_pct:.1f}%")
    print(f"Collaboration accuracy: {100 * collab_acc:.1f}%")
    print(f"Uncertain cases flagged: {n_deferred} ({100 * defer.mean():.1f}%)")
    return {
        "collaboration_accuracy": collab_acc,
        "time_saved_pct": time_saved_pct,
        "workload_reduction": workload_reduction,
    }
