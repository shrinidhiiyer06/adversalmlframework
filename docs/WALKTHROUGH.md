# System Walkthrough

This document narrates an end-to-end experiment run, from data ingestion to adversarial defense.

## Phase 1: Data Preparation

We start by loading the **NSL-KDD** dataset. The `network_loader.py` script cleans the raw text files, encodes categorical features (like protocol type and service), and scales numerical features into a standard range.

## Phase 2: Building the Baseline

Running `scripts/train_baseline.py` initializes a multi-layer perceptron. The model is trained to distinguish between "Normal" traffic and "Attack" traffic based on 41 network features. Once trained, it achieves ≈ 78% accuracy on the test set.

## Phase 3: The Adversarial Attack

In the Red Team scenario, an attacker uses **FGSM** (`whitebox.py`) to perturb a malicious sample. They aim to slightly shift features (like `duration` or `src_bytes`) such that the risk score drops from 0.95 (Highly Suspect) to 0.45 (Benign). Without further defense, this flow would enter the network.

## Phase 4: Zero-Trust Contextual Defense

This is where the system shines. The perturbed flow is intercepted by `zero_trust_engine.py`. Even though the ML model labels it "Benign," the engine enriches the flow with context:

- **Device Trust**: The source IP is coming from a non-MDM enrolled device.
- **Geo-Risk**: The source IP is from a high-risk geographic region.
- **Identity**: No valid session token is attached.

## Phase 5: The Policy Decision

The engine evaluates the flow against its **Priority-Ordered Rules**.

- Rule 1 (Critical Risk) is skipped (score is now 0.45).
- Rule 2 (Untrusted Device) triggers.
**Result**: The decision is escalated from "ALLOW" (from ML) to "DENY" or "STEP_UP_AUTH."

## Phase 6: Telemetry and Reporting

The final decision is logged to `logs/soc_telemetry.json`, and the dashboard updates the **Blue Team** analytics to show a successful adversarial interception. This completes the loop, proving that contextual policies can heal the vulnerabilities of an ML classifier.
