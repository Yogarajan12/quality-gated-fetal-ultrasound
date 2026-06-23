# Diffusion-Based Reconstruction Uncertainty for Quality-Gated Fetal Ultrasound Classification

Selective prediction for fetal ultrasound standard-plane classification. A ResNet-18 classifier is paired with two uncertainty signals — softmax confidence and diffusion reconstruction variance — and an image is auto-accepted or deferred to a human expert based on calibrated thresholds. The repository also reports an honest comparison showing where the diffusion signal helps and where it does not.

> **Status:** research / exploratory. The headline contribution is a working quality-gated deferral pipeline *and* a clear negative result for out-of-domain diffusion priors (see [Results](#results) and [Key Finding](#key-finding--main-limitation)).

---

## Overview

Fetal biometry and screening depend on acquiring correct standard planes, and image quality varies widely with operator and machine. This project asks a narrow, deployment-relevant question: can a model reliably *abstain* on cases it is likely to get wrong, so a sonographer only reviews the uncertain minority?

The pipeline has three stages:

1. **Baseline classifier** — ResNet-18 (ImageNet transfer) over six classes.
2. **Diffusion reconstruction uncertainty** — a pretrained DDPM adds noise to an input and denoises it several times; the per-pixel variance across reconstructions is used as an "unusualness" score.
3. **Quality gate (deferral)** — a case is deferred when softmax confidence is low *or* reconstruction variance is high. Thresholds are calibrated on the validation set for a target coverage.

---

## Method

```
Input image
   │
   ├─► ResNet-18 ──► softmax confidence ─┐
   │                                     ├─► defer if (conf < τ_c) OR (var > τ_u)
   └─► DDPM noise→denoise (MC samples) ──┘        │
              └─► reconstruction variance         ├─► accept ──► model label
                                                  └─► defer  ──► expert review
```

Thresholds `τ_c` (confidence) and `τ_u` (uncertainty) are selected by grid search on the validation set to maximize accuracy on accepted cases at a target coverage (default 85%).

---

## Dataset

[FETAL_PLANES_DB](https://zenodo.org/records/3904280) (Burgos-Artizzu et al., 2020). 12,400 ultrasound images across six classes: *Fetal abdomen, Fetal brain, Fetal femur, Fetal thorax, Maternal cervix, Other*. The split is patient-stratified (≈72.8% train / 13.4% val / 13.8% test) to avoid patient leakage. The download and extraction routine pulls the archive directly from Zenodo at runtime.

---

## Results

**Selective prediction (test set).** The gate trades coverage for accuracy as intended:

| Metric | Value |
|---|---|
| Overall accuracy (no deferral) | 95.50% |
| Coverage (accepted) | 81.52% |
| Deferral rate | 18.48% |
| Accuracy on accepted cases | **99.14%** |
| Accuracy on deferred cases (would-be) | 79.43% |
| Net accuracy gain from deferral | +3.64% |
| Calibrated thresholds | conf 0.882, var 0.0064 |

**Uncertainty-method comparison (error detection).** This is the result that matters most for interpreting the project:

| Method | Acc @ 85% coverage | Error-detection AUC |
|---|---|---|
| Softmax confidence | 99.0% | 0.882 |
| Augmentation ensemble | 99.1% | **0.905** |
| MC Dropout | 97.9% | 0.804 |
| **Diffusion (this work)** | 95.6% | **0.475** |

The diffusion reconstruction variance is the *weakest* of the four signals and sits near chance (0.5) for separating correct from incorrect predictions. The deferral gains reported above are therefore driven almost entirely by the softmax-confidence threshold, not by the diffusion component.

**Out-of-distribution detection.** Consistent with the above, the diffusion signal fails to flag corrupted inputs: detection AUCs are at or below chance for Gaussian noise, blur, low contrast, random, and inverted images (most ≤ 0.5).

**Per-class deferral.** The largest accuracy gains from deferral fall on the hardest classes — *Other* (+8.0%) and *Fetal femur* (+7.3%) — which also carry the highest deferral rates.

**Deployment simulation.** Under the stated assumptions (expert 30 s/image, model 0.5 s/image, expert accuracy 95%), the human-AI workflow reduces expert workload by 81.5% and processing time by 80.2% while holding 98.4% accuracy. These figures depend directly on the assumed expert accuracy and timing and should be read as illustrative.

---

## Key Finding / Main Limitation

The diffusion prior used here is `google/ddpm-celebahq-256` — a face-generation model — standing in for a fetal-ultrasound model that was not available. Reconstruction variance under an **out-of-domain prior** does not encode a meaningful in-domain quality signal, which explains both the near-chance error-detection AUC and the failed OOD detection. The honest takeaway is that *reconstruction-based uncertainty needs a domain-trained diffusion model*; with an off-domain prior, plain softmax confidence and a cheap augmentation ensemble dominate. Training or fine-tuning a fetal-US diffusion model is the single clearest next step (see [Roadmap](#roadmap)).

---

## Repository Structure

The current source of truth is the notebook. A suggested refactor into an importable package:

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── configs/
│   └── default.yaml            # paths, image size, classes, thresholds
├── notebooks/
│   └── fetal_us_uncertainty.ipynb
├── src/
│   ├── data.py                 # dataset, transforms, dataloaders
│   ├── classifier.py           # ResNet-18 model + training loop
│   ├── diffusion_uncertainty.py# noise/denoise + variance estimator
│   ├── gating.py               # QualityGatedClassifier, calibration
│   ├── evaluation.py           # metrics, plots, OOD, deployment sim
│   └── baselines.py            # MC Dropout, augmentation ensemble
├── scripts/
│   └── run_experiment.py       # main() entrypoint
├── results/
│   ├── final_metrics.csv
│   └── *_uncertainties.npy      # cached, optional
└── assets/                     # figures referenced by this README
```

---

## Installation

```bash
git clone https://github.com/<user>/<repo>.git
cd <repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A single L4/T4 GPU is sufficient; the workload is inference-heavy with minimal training.

---

## Usage

```bash
python scripts/run_experiment.py --config configs/default.yaml
```

The run downloads the dataset, trains the baseline classifier, computes (or loads cached) uncertainties, calibrates thresholds, evaluates the gate, and writes figures and `final_metrics.csv`.

> **Note on paths:** the original notebook hardcodes Deepnote-specific paths (`/work/...`, `/tmp/fetal_data/...`). These should be moved into `configs/default.yaml` before the code runs portably elsewhere.

---

## Reproducibility

A global seed (42) is set for Python, NumPy, and PyTorch. Diffusion sampling remains stochastic across the Monte Carlo reconstructions, so uncertainty values vary slightly between runs; cached `.npy` uncertainties are loaded when present to keep evaluation deterministic.

---

## Roadmap

- Train or fine-tune a **fetal-ultrasound diffusion model** and re-run the comparison; this is the experiment that would make reconstruction uncertainty competitive.
- Add **temperature scaling** so the confidence signal is calibrated, then re-derive thresholds.
- Report **selective-risk / risk-coverage curves with confidence intervals** rather than point estimates.
- Externalize all configuration; remove environment-specific paths.
- Add a short **model card** and **data-use statement** given the clinical framing.

---

## Citation

If this dataset is used, please cite the original authors:

```bibtex
@article{burgos2020fetalplanes,
  title   = {Evaluation of deep convolutional neural networks for automatic
             classification of common maternal fetal ultrasound planes},
  author  = {Burgos-Artizzu, Xavier P. and Coronado-Guti{\'e}rrez, David and
             Valenzuela-Alcaraz, Brenda and Bonet-Carne, Elisenda and
             Eixarch, Elisenda and Crispi, Fatima and Gratac{\'o}s, Eduard},
  journal = {Scientific Reports},
  volume  = {10},
  pages   = {10200},
  year    = {2020}
}
```

---

## License

Released under the MIT License (see `LICENSE`). The FETAL_PLANES_DB dataset is governed by its own license on Zenodo; review it before redistribution.

## Acknowledgements

Built with PyTorch, Hugging Face `diffusers`, and scikit-learn. Dataset courtesy of BCNatal / Burgos-Artizzu et al.

---

*Research code for methodological study only. Not a medical device and not validated for clinical use.*
