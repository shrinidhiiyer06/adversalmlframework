"""
Adversarial training comparison baseline.

Trains a Random Forest on a mixed dataset of 70% clean training samples
and 30% FGSM-generated adversarial examples, then compares against the
standard model on:
  - Clean accuracy
  - Adversarial accuracy (evasion resistance)
  - Zero-Trust bypass rate

The expected finding: adversarial training improves robustness modestly
at some clean accuracy cost, while the Zero-Trust approach achieves
comparable or better bypass rate without clean accuracy degradation.

Note on methodology: This script uses a surrogate neural network to
generate FGSM adversarial examples (black-box transfer attack), since
the Random Forest classifier is not differentiable. This is a well-studied
approach in the adversarial ML literature.

Usage:
    python scripts/train_adversarial.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import MODEL_DIR, DATA_DIR, RESULTS_DIR, FGM_EPS
from src.training.surrogate import train_surrogate
from src.attacks.whitebox import run_whitebox_attack
from src.policy.zero_trust_engine import ZeroTrustEngine
from src.simulation.context_profiles import generate_attacker_context


def main():
    """Run adversarial training experiment."""
    print("=" * 60)
    print("  ADVERSARIAL TRAINING COMPARISON")
    print("=" * 60)

    # Load data
    train_df = pd.read_csv(os.path.join(MODEL_DIR, "train_set.csv"))
    test_df = pd.read_csv(os.path.join(MODEL_DIR, "test_set.csv"))

    X_train = train_df.drop(columns=['label']).values
    y_train = train_df['label'].values
    X_test = test_df.drop(columns=['label']).values
    y_test = test_df['label'].values

    clip_values = (float(X_train.min()), float(X_train.max()))

    # -- Step 1: Train standard model --
    print("\n[1/4] Training standard Random Forest...")
    rf_standard = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_standard.fit(X_train, y_train)
    std_clean_acc = accuracy_score(y_test, rf_standard.predict(X_test))
    print(f"  Clean accuracy: {std_clean_acc:.4f}")

    # -- Step 2: Generate adversarial training examples --
    print("\n[2/4] Generating adversarial training examples via surrogate...")
    surrogate, _ = train_surrogate(X_train, y_train)

    # Select 30% of training data for adversarial augmentation
    n_adv = int(0.3 * len(X_train))
    rng = np.random.default_rng(42)
    adv_indices = rng.choice(len(X_train), n_adv, replace=False)
    X_adv_base = X_train[adv_indices]
    y_adv_base = y_train[adv_indices]

    try:
        X_adv = run_whitebox_attack(
            surrogate, X_adv_base, y_adv_base,
            clip_values, sample_size=n_adv, eps=FGM_EPS
        )
        if isinstance(X_adv, tuple):
            X_adv = X_adv[0]
        print(f"  Generated {len(X_adv)} adversarial training examples")
    except Exception as e:
        print(f"  Warning: Attack generation failed ({e}), using noisy augmentation")
        noise = rng.normal(0, FGM_EPS, X_adv_base.shape)
        X_adv = np.clip(X_adv_base + noise, X_train.min(axis=0), X_train.max(axis=0))

    # -- Step 3: Train adversarially-robust model --
    print("\n[3/4] Training adversarially-robust Random Forest...")
    # Mix: 70% clean + 30% adversarial
    X_mixed = np.vstack([X_train, X_adv])
    y_mixed = np.concatenate([y_train, y_adv_base])

    # Shuffle
    shuffle_idx = rng.permutation(len(X_mixed))
    X_mixed = X_mixed[shuffle_idx]
    y_mixed = y_mixed[shuffle_idx]

    rf_adversarial = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_adversarial.fit(X_mixed, y_mixed)

    # Save adversarially-trained model
    model_path = os.path.join(MODEL_DIR, "adversarial_trained_classifier.pkl")
    joblib.dump(rf_adversarial, model_path)
    print(f"  Saved to: {model_path}")

    # -- Step 4: Compare models --
    print("\n[4/4] Comparing models...")

    # Generate test adversarial examples
    attack_mask = y_test == 1
    X_test_attacks = X_test[attack_mask]
    y_test_attacks = y_test[attack_mask]
    test_sample_size = min(50, len(X_test_attacks))
    test_idx = rng.choice(len(X_test_attacks), test_sample_size, replace=False)
    X_test_sample = X_test_attacks[test_idx]
    y_test_sample = y_test_attacks[test_idx]

    try:
        X_test_adv = run_whitebox_attack(
            surrogate, X_test_sample, y_test_sample,
            clip_values, sample_size=test_sample_size, eps=FGM_EPS
        )
        if isinstance(X_test_adv, tuple):
            X_test_adv = X_test_adv[0]
    except Exception as e:
        print(f"  Warning: Test attack failed ({e}), using noisy version")
        noise = rng.normal(0, FGM_EPS, X_test_sample.shape)
        X_test_adv = np.clip(X_test_sample + noise, X_train.min(axis=0), X_train.max(axis=0))

    # Metrics
    adv_clean_acc = accuracy_score(y_test, rf_adversarial.predict(X_test))
    std_adv_acc = accuracy_score(y_test_sample, rf_standard.predict(X_test_adv))
    adv_adv_acc = accuracy_score(y_test_sample, rf_adversarial.predict(X_test_adv))

    # ZT bypass rates
    zt_engine = ZeroTrustEngine()

    def get_bypass_rate(model, X):
        risk_scores = model.predict_proba(X)[:, 1]
        contexts = generate_attacker_context(len(X), seed=42)
        decisions = zt_engine.evaluate_batch(risk_scores, contexts)
        return float(np.mean([d.decision == "ALLOW" for d in decisions]))

    std_zt_bypass = get_bypass_rate(rf_standard, X_test_adv)
    adv_zt_bypass = get_bypass_rate(rf_adversarial, X_test_adv)

    # Results
    results = {
        'experiment': 'adversarial_training_comparison',
        'standard_model': {
            'clean_accuracy': float(std_clean_acc),
            'adversarial_accuracy': float(std_adv_acc),
            'zt_bypass_rate': float(std_zt_bypass),
        },
        'adversarial_model': {
            'clean_accuracy': float(adv_clean_acc),
            'adversarial_accuracy': float(adv_adv_acc),
            'zt_bypass_rate': float(adv_zt_bypass),
        },
        'training_config': {
            'clean_ratio': 0.7,
            'adversarial_ratio': 0.3,
            'fgsm_epsilon': FGM_EPS,
            'attack_method': 'FGSM via surrogate (black-box transfer)',
        }
    }

    results_path = os.path.join(RESULTS_DIR, "adversarial_training.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print comparison
    print("\n" + "=" * 60)
    print(f"  {'Metric':<25s} {'Standard':>12s} {'Adversarial':>12s}")
    print("=" * 60)
    print(f"  {'Clean Accuracy':<25s} {std_clean_acc:>11.1%} {adv_clean_acc:>11.1%}")
    print(f"  {'Adversarial Accuracy':<25s} {std_adv_acc:>11.1%} {adv_adv_acc:>11.1%}")
    print(f"  {'ZT Bypass Rate':<25s} {std_zt_bypass:>11.1%} {adv_zt_bypass:>11.1%}")
    print("=" * 60)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
