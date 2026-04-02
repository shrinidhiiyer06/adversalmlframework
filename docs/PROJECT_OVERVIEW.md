# Project Overview

## Summary

The **Zero-Trust Adversarial Intrusion Detection System** is a research-driven project that explores the intersection of Machine Learning security and Zero-Trust Network Architecture (ZTNA). It demonstrates how contextual policy layers can effectively mitigate adversarial evasion attacks against ML models without requiring expensive and often brittle adversarial retraining.

## Objectives

1. **Quantify Vulnerability**: Measure how easily standard ML-based NIDS can be fooled by gradient-based attacks (FGSM/PGD).
2. **Implement Zero-Trust Defense**: Build a modular system that combines ML inference with multi-factor contextual signals (Device, Location, Identity).
3. **Validate Robustness**: Prove that the "system-level" accuracy remains high even when individual component (ML) accuracy drops due to adversarial pressure.

## Key Findings

- **Machine Learning is fragile**: A small perturbation (ε=0.05) can achieve a 20% evasion rate.
- **Context is king**: Adding even a single layer of context (e.g., Device Trust) cuts the success rate of attacks significantly.
- **Full Zero-Trust is resilient**: A multi-factor configuration reduced the effective bypass rate to **0%** in our experiments on the NSL-KDD benchmark.

## Contributions

- A modular **Adversarial Attack Engine** for network traffic data.
- A functional **Priority-Ordered Zero-Trust Policy Engine**.
- A real-time **SOC Telemetry Dashboard** for evaluating adversarial robustness.
- Rigorous statistical evaluation (p-value < 0.001) of Zero-Trust as a defense mechanism.

---
*SRM Institute of Science and Technology · Final Year Research Project*
