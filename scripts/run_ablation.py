"""
Ablation study script for Zero-Trust policy layer evaluation.

Evaluates four system configurations to isolate the contribution of each
contextual factor:
    1. ML classifier only (no Zero-Trust context)
    2. ML + device trust context only
    3. ML + geo-risk context only
    4. Full system (ML + device trust + geo-risk + time-of-day + identity)

For each configuration, reports:
    - Deny rate on adversarial flows (FGSM-perturbed evasion samples)
    - False positive rate on legitimate traffic (using FULL label=0 test set)
    - Effective bypass rate

The key insight: adversarial EVASION means the attacker has reduced the ML
risk score via FGSM perturbation. The ablation shows that even when ML is
fooled, contextual policies still deny the flow.

Usage:
    python scripts/run_ablation.py
"""

import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import (
    MODEL_DIR, DATA_DIR, RESULTS_DIR,
    DEMO_SAMPLE_COUNT
)
from src.data.network_loader import NetworkDataLoader
from src.policy.zero_trust_engine import ZeroTrustEngine
from src.risk_engine.network_classifier import NetworkRiskClassifier
from src.attacks.network_adversarial import NetworkAdversarialAttacker
from src.simulation.context_profiles import (
    generate_attacker_context, generate_legitimate_context
)


def load_assets():
    """Load model, preprocessors, and NSL-KDD test data."""
    # Load neural network risk classifier
    nn_model = NetworkRiskClassifier(input_dim=41)
    nn_model.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "network_risk_classifier.pth"),
        map_location='cpu', weights_only=True
    ))
    nn_model.eval()

    # Load NSL-KDD test data using the same preprocessing as training
    loader = NetworkDataLoader()
    loader.load_preprocessors(MODEL_DIR)

    test_path = os.path.join(DATA_DIR, 'KDDTest+.txt')
    X_test, y_test, _ = loader.load_and_preprocess(test_path, is_train=False)

    # Compute feature bounds from test data for domain-constrained attacks
    feature_bounds = loader.get_feature_bounds(X_test)

    return nn_model, feature_bounds, X_test, y_test


def generate_adversarial_evasion_samples(
    nn_model, feature_bounds, X_test, y_test,
    epsilon=0.05, n_samples=100, seed=42
):
    """Generate FGSM adversarial evasion samples.

    Takes attack-labeled samples, applies FGSM to reduce the neural net's
    risk score, then measures evasion success.

    Returns:
        Dict with adversarial samples, clean/perturbed risk scores, and
        evasion statistics.
    """
    rng = np.random.default_rng(seed)

    # Select attack-labeled samples
    attack_mask = y_test == 1
    X_attacks = X_test[attack_mask]

    if len(X_attacks) > n_samples:
        indices = rng.choice(len(X_attacks), n_samples, replace=False)
        X_selected = X_attacks[indices]
    else:
        X_selected = X_attacks
        n_samples = len(X_selected)

    # Score clean samples with neural net
    with torch.no_grad():
        clean_scores = nn_model.predict_batch(X_selected.astype(np.float32))

    # Create adversarial attacker and generate FGSM examples
    attacker = NetworkAdversarialAttacker(nn_model, feature_bounds)

    X_adv_list = []
    for i in range(n_samples):
        x_adv = attacker.constrained_fgsm(
            X_selected[i], epsilon=epsilon, target_label=0
        )
        X_adv_list.append(x_adv[0])

    X_adv = np.array(X_adv_list)

    # Score adversarial samples (post-attack risk scores)
    with torch.no_grad():
        adv_scores = nn_model.predict_batch(X_adv.astype(np.float32))

    # Evasion = adversarial score dropped below 0.5
    evasion_mask = adv_scores < 0.5

    return {
        'X_clean': X_selected,
        'X_adv': X_adv,
        'clean_scores': clean_scores,
        'adv_scores': adv_scores,
        'evasion_rate': float(evasion_mask.mean()),
        'n_samples': n_samples,
        'epsilon': epsilon,
    }


def run_single_config(config_name, engine, adv_scores, adv_contexts,
                      legit_scores, legit_contexts):
    """Evaluate a single ablation configuration."""
    n_adv = len(adv_scores)
    n_legit = len(legit_scores)

    # Evaluate adversarial samples (FGSM-perturbed)
    adv_decisions = engine.evaluate_batch(adv_scores, adv_contexts)
    n_denied_adv = sum(1 for d in adv_decisions if d.decision == "DENY")
    deny_rate = n_denied_adv / n_adv if n_adv > 0 else 0.0
    bypass_rate = 1.0 - deny_rate

    # Evaluate legitimate samples (FPR)
    legit_decisions = engine.evaluate_batch(legit_scores, legit_contexts)
    n_denied_legit = sum(1 for d in legit_decisions if d.decision == "DENY")
    fpr = n_denied_legit / n_legit if n_legit > 0 else 0.0

    # Per-sample details
    per_sample = []
    for i, dec in enumerate(adv_decisions):
        per_sample.append({
            'sample_idx': i,
            'ml_risk': float(adv_scores[i]),
            'ml_decision': 'DENY' if adv_scores[i] > 0.5 else 'ALLOW',
            'zt_decision': dec.decision,
            'rule_fired': dec.rule_fired,
            'device_trust': adv_contexts[i]['device_trust'],
            'geo_risk': adv_contexts[i]['geo_risk'],
        })

    return {
        'config': config_name,
        'n_adversarial': n_adv,
        'n_legitimate': n_legit,
        'deny_rate': float(deny_rate),
        'false_positive_rate': float(fpr),
        'bypass_rate': float(bypass_rate),
        'n_denied_adversarial': n_denied_adv,
        'n_denied_legitimate': n_denied_legit,
        'per_sample_details': per_sample,
    }


def main():
    """Run the full ablation study."""
    print("=" * 60)
    print("  ZERO-TRUST ABLATION STUDY (with FGSM Adversarial Evasion)")
    print("=" * 60)

    # Load assets
    nn_model, feature_bounds, X_test, y_test = load_assets()

    # Step 1: Generate FGSM adversarial evasion samples
    print("\n[1/3] Generating FGSM adversarial evasion samples (eps=0.05)...")
    evasion = generate_adversarial_evasion_samples(
        nn_model, feature_bounds, X_test, y_test,
        epsilon=0.05, n_samples=DEMO_SAMPLE_COUNT * 4,  # 120 samples
        seed=42
    )
    print(f"  Generated {evasion['n_samples']} adversarial samples")
    print(f"  Evasion rate (score < 0.5): {evasion['evasion_rate']:.1%}")
    print(f"  Avg risk: {evasion['clean_scores'].mean():.3f} -> "
          f"{evasion['adv_scores'].mean():.3f}")

    # Save demo samples
    demo_path = os.path.join(DATA_DIR, "demo_samples.npy")
    np.save(demo_path, evasion['X_adv'][:DEMO_SAMPLE_COUNT])
    print(f"  Saved demo samples to: {demo_path}")

    # Step 2: Prepare contexts and legitimate samples
    print("\n[2/3] Preparing evaluation data...")
    n_adv = evasion['n_samples']
    adv_scores = evasion['adv_scores']  # post-FGSM neural net scores
    adv_contexts = generate_attacker_context(n_adv, seed=42)

    legit_mask = y_test == 0
    X_legit = X_test[legit_mask]
    with torch.no_grad():
        legit_scores = nn_model.predict_batch(X_legit.astype(np.float32))
    legit_contexts = generate_legitimate_context(len(X_legit), seed=1042)
    print(f"  Using {n_adv} adversarial + {len(X_legit)} legitimate samples")

    # Step 3: Run ablation
    print("\n[3/3] Running ablation study...")
    configs = ZeroTrustEngine.ABLATION_CONFIGS.items()

    all_results = []
    for config_name, factors in configs:
        print(f"\n{'_'*50}")
        print(f"  Config: {config_name}")
        print(f"  Enabled: {factors if factors else '(ML only)'}")
        print(f"{'_'*50}")

        engine = ZeroTrustEngine(enabled_factors=factors)
        result = run_single_config(
            config_name, engine, adv_scores, adv_contexts,
            legit_scores, legit_contexts
        )
        all_results.append(result)

        print(f"  Deny Rate (adversarial):    {result['deny_rate']:.1%}")
        print(f"  False Positive Rate:        {result['false_positive_rate']:.1%}")
        print(f"  Effective Bypass Rate:      {result['bypass_rate']:.1%}")

    # Save results
    results_payload = {
        'experiment': 'zero_trust_ablation',
        'attack_method': 'FGSM',
        'epsilon': 0.05,
        'n_adversarial_samples': n_adv,
        'n_legitimate_samples': len(X_legit),
        'evasion_rate': evasion['evasion_rate'],
        'configurations': [
            {k: v for k, v in r.items() if k != 'per_sample_details'}
            for r in all_results
        ],
        'detailed_results': all_results,
    }

    results_path = os.path.join(RESULTS_DIR, "ablation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results_payload, f, indent=2)

    # Print summary table
    print("\n" + "=" * 70)
    print(f"  {'Configuration':<30s} {'Deny%':>8s} {'FPR%':>8s} {'Bypass%':>8s}")
    print("=" * 70)
    for r in all_results:
        print(f"  {r['config']:<30s} {r['deny_rate']:>7.1%} "
              f"{r['false_positive_rate']:>7.1%} {r['bypass_rate']:>7.1%}")
    print("=" * 70)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
