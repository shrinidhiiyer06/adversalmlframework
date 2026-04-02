# Usage Guide

This guide explains how to interact with the Zero-Trust Adversarial IDS through its dashboard and CLI scripts.

## Getting Started

Follow the Quick Start steps in the [README](../README.md) to install dependencies and train the baseline model.

## Interactive Dashboard

Launch the dashboard with:

```bash
streamlit run src/dashboard/app.py
```

### Dashboard Tabs

1. **SOC Console (Live Traffic)**
   - Simulate live network traffic flowing through the system.
   - Observe real-time ML risk scores and final Zero-Trust policy decisions.
   - Watch how "Benign" or "Malicious" labels are handled by different rules.

2. **Red Team (Attack Testing)**
   - Manually trigger adversarial attacks (FGSM, PGD) against specific network samples.
   - Compare the original risk score with the adversarially suppressed score.
   - See how the Zero-Trust layer catches attacks that bypass the ML model.

3. **Blue Team (Defense Analytics)**
   - View high-level metrics: Accuracy, Precision, Recall.
   - Analyze the "Evasion Gap" — where the ML model was fooled but the system remained secure.
   - Visualizations of the system's resilience across different configurations.

## CLI Scripts for Reproducibility

For automated experiments and reproducing paper results:

- `python scripts/run_ablation.py`: Executes the 4-configuration study.
- `python scripts/run_epsilon_sweep.py`: Analyzes system performance across a range of attack intensities.

Results are stored in the `results/` folder as JSON objects for further analysis.
