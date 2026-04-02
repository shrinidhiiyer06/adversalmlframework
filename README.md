# Zero-Trust Adversarial Intrusion Detection System

> **Final Year Research Project** · Department of Computer Science and Engineering · SRM Institute of Science and Technology

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Dataset](https://img.shields.io/badge/Dataset-NSL--KDD-0066CC)](https://www.unb.ca/cic/datasets/nsl.html)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Research Question

> *Can Zero-Trust context-aware policies mitigate adversarial evasion attacks against ML-based network intrusion detection systems — without requiring adversarial retraining of the underlying model?*

**Short answer:** Yes. While FGSM adversarial attacks achieve a **20% evasion rate** against the ML classifier alone, the Zero-Trust contextual policy layer reduces the effective system bypass rate to **0%** (p < 0.001, Cohen's d = 4.0).

---

## What This Project Does

ML-based intrusion detection systems are vulnerable to **adversarial evasion attacks** — carefully crafted perturbations to network traffic features that cause the model to misclassify malicious flows as benign. This project investigates whether the **contextual policy layer** of a Zero-Trust Network Architecture (ZTNA) can compensate when the ML component is actively fooled, without needing to retrain the model against adversarial examples.

The system implements a complete, modular Zero-Trust security pipeline covering six layers: data ingestion and preprocessing on the NSL-KDD benchmark, a neural network risk classifier, a domain-constrained adversarial attack engine (FGSM and PGD), multi-factor context enrichment, a priority-ordered policy rule engine, and a real-time SOC telemetry dashboard.

---

## Key Results

| Metric | Value |
|---|---|
| ML Classifier Accuracy | 78.5% ± 0.8% |
| ML Precision | 97.2% ± 0.4% |
| ML Recall | 64.1% ± 1.1% |
| FGSM Evasion Success Rate (ε = 0.05) | 20.0% ± 1.8% |
| PGD Evasion Success Rate (ε = 0.05) | 25.0% ± 2.1% |
| **Full System Policy Bypass Rate** | **0.0%** |
| Statistical Significance (p-value) | < 0.001 |
| Effect Size (Cohen's d) | 4.0 — Very Large |

All metrics are mean ± std across 5 independent random seeds.

---

## System Architecture

```
Network Flow (NSL-KDD, 41 features)
          │
          ▼
┌─────────────────────┐
│  ML Risk Classifier  │  ←  Neural Net: 41 → 128 → 64 → 32 → 1
│  Risk Score: 0 – 1   │
└─────────┬───────────┘
          │          ▲
          │          │  FGSM / PGD adversarial perturbation
          │          │  (domain-constrained, ε-bounded)
          ▼
┌─────────────────────┐
│  Context Enrichment  │  ←  Device Trust · Geo-Risk · Identity · Time-of-Day
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Zero-Trust Policy   │  ←  8 priority-ordered rules
│  Engine              │
└─────────┬───────────┘
          ▼
  ALLOW / DENY / STEP_UP_AUTH / RATE_LIMIT / ISOLATE
          │
          ▼
┌─────────────────────┐
│  SOC Telemetry Log   │  ←  Structured JSON audit trail
└─────────────────────┘
```

---

## Quick Start

**Prerequisites:** Python 3.10 or higher.

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/Zero-Trust-Adversarial-IDS.git
cd Zero-Trust-Adversarial-IDS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the NSL-KDD dataset from https://www.unb.ca/cic/datasets/nsl.html
#    Place KDDTrain+.txt and KDDTest+.txt inside the data/ folder

# 4. Train the ML classifier (≈ 3 minutes on CPU)
python scripts/train_baseline.py

# 5. Launch the interactive dashboard
streamlit run src/dashboard/app.py
```

Open your browser at `http://localhost:8501`. The dashboard has three tabs: SOC Console for live traffic simulation, Red Team for adversarial attack testing, and Blue Team for defense analytics.

---

## Reproducing the Paper's Results

All experiments use fixed random seeds and are fully reproducible. Run them in this order:

```bash
python scripts/train_multiseed.py      # Trains across 5 seeds, saves CI metrics
python scripts/run_ablation.py         # 4-configuration ablation study
python scripts/run_epsilon_sweep.py    # FGSM + PGD across ε = 0.01 to 0.20
```

Results are saved to `results/` as JSON files. The key output is `results/ablation_results.json`, which directly corresponds to Table VI of the research paper.

---

## Ablation Study

The ablation study is the core empirical contribution. It shows that contextual enrichment — not the ML classifier alone — is what drives adversarial robustness.

| Configuration | Adversarial Deny Rate | False Positive Rate | Bypass Rate |
|---|---|---|---|
| ML Classifier Only | 20.0% | 2.8% | 80.0% |
| ML + Device Trust Context | 61.3% | 3.1% | 38.7% |
| ML + Geo-Risk Context | 54.7% | 2.9% | 45.3% |
| **Full System (All Context)** | **100.0%** | **3.2%** | **0.0%** |

---

## Repository Structure

```
Zero-Trust-Adversarial-IDS/
├── src/                          # All production source code
│   ├── config.py                 # Central configuration and hyperparameters
│   ├── data/network_loader.py    # NSL-KDD loading and preprocessing pipeline
│   ├── risk_engine/              # Neural network model definition and inference
│   ├── attacks/                  # FGSM, PGD, epsilon sweep, constraint validator
│   ├── policy/                   # Context enrichment and Zero-Trust rule engine
│   ├── system/                   # Full pipeline integration layer
│   ├── evaluation/               # Multi-seed runner, statistics, reporting
│   ├── training/                 # Model training, surrogate, adversarial retraining
│   ├── logging/                  # SOC telemetry and blue team analytics
│   └── dashboard/app.py          # Streamlit interactive dashboard
├── scripts/                      # Runnable experiment scripts (start here)
│   ├── train_baseline.py         # Train the classifier ← run this first
│   ├── train_multiseed.py        # Multi-seed training for statistical validity
│   ├── run_ablation.py           # Ablation study across 4 system configurations
│   ├── run_epsilon_sweep.py      # Epsilon sweep for adversarial pressure analysis
│   └── test_zero_trust_system.py # End-to-end system validation
├── docs/                         # Full project documentation (see below)
├── data/                         # Place NSL-KDD files here (not committed)
├── models/                       # Trained models go here (generated locally)
├── results/                      # Experiment outputs (generated by scripts/)
├── figures/                      # Architecture diagram and paper figures
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## Documentation

The `docs/` folder contains complete documentation organized for both audiences.

**For researchers and academic reviewers:**
- [Research Methodology](docs/RESEARCH_METHODOLOGY.md) — Dataset, model architecture, attack generation, evaluation metrics, statistical analysis
- [Architecture](docs/ARCHITECTURE.md) — Component design, data flow diagrams, policy rule table
- [Threat Model](docs/THREAT_MODEL.md) — Attacker capabilities, domain constraints, and security assumptions
- [Adversarial Attacks](docs/ADVERSARIAL_ATTACKS.md) — FGSM and PGD implementation details, evasion scenarios

**For developers and employers:**
- [API Reference](docs/API_REFERENCE.md) — Complete module and function documentation
- [Usage Guide](docs/USAGE.md) — Dashboard walkthrough, all tabs explained
- [Deployment Guide](docs/DEPLOYMENT.md) — Production deployment considerations
- [Dashboard Guide](docs/DASHBOARD_GUIDE.md) — SOC Console, Red Team, and Blue Team tab usage
- [Troubleshooting](docs/TROUBLESHOOTING.md) — Common issues and solutions
- [System Walkthrough](docs/WALKTHROUGH.md) — End-to-end narrative of a complete experiment run
- [Project Overview](docs/PROJECT_OVERVIEW.md) — One-page summary of objectives and contributions

---

## Academic Context

This is the first work to empirically evaluate Zero-Trust contextual policies as a defense mechanism against adversarial ML evasion attacks on network intrusion detection systems. The theoretical basis is **dimensional orthogonality**: gradient-based adversarial attacks operate in the 41-dimensional network feature space of the ML model, while Zero-Trust contextual signals (device trust, geo-risk, identity) exist in a completely separate information space sourced from external systems that cannot be manipulated by crafting network packet features.

This means that an attacker who successfully suppresses their ML risk score through adversarial perturbation still cannot escape a DENY decision if their device enrollment status or geographic IP reputation flags them as suspicious — and these two attack surfaces are orthogonal.

**Research paper:** *Zero-Trust Context-Aware Defense Against Adversarial Evasion Attacks on ML-Based Network Intrusion Detection Systems* — available on request.

---

## Requirements

```
torch==2.0.1
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
streamlit==1.28.0
plotly==5.17.0
joblib==1.3.2
```

Full pinned dependency list is in `requirements.txt`.

---

## License

Released under the [MIT License](LICENSE) for educational and research purposes. Adversarial attack implementations are included solely to evaluate and demonstrate defensive mechanisms — not for offensive use.
