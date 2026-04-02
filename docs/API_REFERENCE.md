# API Reference

This document provides a summary of the core modules and functions in the `Zero-Trust-Adversarial-IDS` project.

## Core Modules

### `src/attacks/` - Adversarial Attack Engine

- `whitebox.py`: Implementation of FGSM (Fast Gradient Method) and PGD (Projected Gradient Descent).
- `blackbox.py`: Implementation of HopSkipJump black-box attacks.
- `validate_constraints.py`: Logic to ensure adversarial perturbations result in valid network flows.
- `network_adversarial.py`: High-level wrapper for generating adversarial datasets.

### `src/policy/` - Zero-Trust Rule Engine

- `zero_trust_engine.py`: The core priority-ordered rule engine that combines ML risk scores with contextual signals.
- `network_context.py`: Providers for device trust, geo-risk, and identity context.
- `trust_model.py`: Definitions for device trust levels and risk scoring components.

### `src/risk_engine/` - ML Classifier

- `model.py`: Neural network architecture definition (PyTorch).
- `inference.py`: Functions for getting risk scores and labels from trained models.

### `src/evaluation/` - Performance & Statistics

- `statistics.py`: Calculation of mean ± std across multiple seeds, p-values, and effect sizes (Cohen's d).
- `reporting.py`: JSON and Markdown report generation for ablation studies and epsilon sweeps.
- `runner.py`: Orchestrator for running batch experiments.

### `src/data/` - Data Pipeline

- `network_loader.py`: Loading, cleaning, and preprocessing the NSL-KDD dataset.

## Key Functions

### System Integration

`src/system/zero_trust_network.py`

- `simulate_traffic_flow(features, context)`: Processes a single network flow through the entire ML + ZT pipeline.

### Experimentation

`scripts/run_ablation.py`

- Runs the 4-configuration study: ML only, ML+Device, ML+Geo, and Full System.

For a narrative walkthrough of how these components interact, see the [System Walkthrough](WALKTHROUGH.md).
