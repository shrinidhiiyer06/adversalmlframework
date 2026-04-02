# Reproducibility Guide

## Prerequisites

- Python 3.8+
- All packages from `requirements.txt` installed: `pip install -r requirements.txt`

## Quick Start (Full Pipeline)

```bash
# For Windows users:
scripts\run_all_experiments.bat

# For Mac/Linux users (once a .sh version is created):
# ./scripts/run_all_experiments.sh
```

Or run each step manually:

## Step-by-Step Reproduction

### 1. Generate Training Data

```bash
python src/simulation/traffic_generator.py
python src/simulation/attack_generator.py
```

### 2. Train Baseline Models

```bash
python src/training/core_model.py
```

### 3. Multi-Seed Evaluation (Section 3.1)

```bash
python scripts/train_multiseed.py
# Output: results/multiseed_results.json, models/seed_*/
```

### 4. Adversarial Training Comparison (Section 3.2)

```bash
python scripts/train_adversarial.py
# Output: results/adversarial_training.json, models/adversarial_trained_classifier.pkl
```

### 5. ROC & Threshold Analysis (Section 3.3)

```bash
python scripts/generate_roc.py
# Output: results/roc_threshold.json, figures/roc_curve.png, figures/confusion_matrix.png
```

### 6. Epsilon Sweep (Section 4.1)

```bash
python scripts/run_epsilon_sweep.py
# Output: results/epsilon_sweep.json, figures/epsilon_sweep.png
```

### 7. Zero-Trust Ablation Study (Section 5.1)

```bash
python scripts/run_ablation.py
# Output: results/ablation_results.json, data/demo_samples.npy
```

### 8. Launch Dashboard with Research Demo

```bash
streamlit run src/dashboard/app.py
# Navigate to "Research Demo" tab
```

## Expected Outputs

| File | Description |
| --- | --- |
| `results/multiseed_results.json` | 5-seed aggregate stats with 95% CIs |
| `results/adversarial_training.json` | Standard vs adversarial-trained model comparison |
| `results/roc_threshold.json` | ROC-AUC and threshold sweep metrics |
| `results/epsilon_sweep.json` | FGSM/PGD at 5 epsilon values |
| `results/ablation_results.json` | 4-config Zero-Trust ablation |
| `figures/roc_curve.png` | Figure 2: ROC curve |
| `figures/confusion_matrix.png` | Figure 5: Confusion matrix |
| `figures/epsilon_sweep.png` | Figure 3: Epsilon sweep chart |

## Model Checkpoints

Per-seed models are saved to `models/seed_{0,42,123,456,789}/`. Each contains:

- `random_forest.pkl`
- `isolation_forest.pkl`

## Notes

- All random seeds are fixed for deterministic reproduction
- The CICIDS-2017 dataset requires manual download from UNB (see `src/data/cicids_loader.py`)
- Results may vary slightly across NumPy/scikit-learn versions
