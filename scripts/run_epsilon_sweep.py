"""
Epsilon sweep experiment for adversarial robustness evaluation.

Sweeps FGSM and PGD at epsilon = [0.01, 0.02, 0.05, 0.10, 0.20] and records:
  - Evasion success rate
  - Robust accuracy
  - Average L2 perturbation magnitude
  - Average L-infinity perturbation magnitude

Results saved to results/epsilon_sweep.json and figures generated.

Usage:
    python scripts/run_epsilon_sweep.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import (
    MODEL_DIR, DATA_DIR, RESULTS_DIR, FIGURES_DIR,
    EPS_VALUES, PGD_ITERATIONS, PGD_ALPHA_FACTOR, PGD_RESTARTS
)
from src.attacks.whitebox import run_whitebox_attack
from src.attacks.validate_constraints import validate_adversarial_batch
from src.training.surrogate import train_surrogate
from src.policy.zero_trust_engine import ZeroTrustEngine
from src.simulation.context_profiles import generate_attacker_context


def compute_perturbation_norms(X_orig, X_adv):
    """Compute L2 and L-inf perturbation norms.

    Args:
        X_orig: Original samples.
        X_adv: Adversarial samples.

    Returns:
        Tuple of (mean_l2, mean_linf).
    """
    diff = X_adv - X_orig
    l2_norms = np.linalg.norm(diff, axis=1)
    linf_norms = np.max(np.abs(diff), axis=1)
    return float(np.mean(l2_norms)), float(np.mean(linf_norms))


def main():
    """Run the full epsilon sweep experiment."""
    print("=" * 60)
    print("  EPSILON SWEEP EXPERIMENT")
    print(f"  Epsilons: {EPS_VALUES}")
    print("=" * 60)

    # Load assets
    rf = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
    iso = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.pkl"))
    test_df = pd.read_csv(os.path.join(MODEL_DIR, "test_set.csv"))
    train_df = pd.read_csv(os.path.join(MODEL_DIR, "train_set.csv"))

    X_test = test_df.drop(columns=['label']).values
    y_test = test_df['label'].values
    X_train = train_df.drop(columns=['label']).values
    y_train = train_df['label'].values

    # Feature bounds for clipping
    feature_min = X_train.min(axis=0)
    feature_max = X_train.max(axis=0)
    clip_values = (float(feature_min.min()), float(feature_max.max()))

    # Select attack samples (only malicious samples)
    attack_mask = y_test == 1
    X_attacks = X_test[attack_mask]
    y_attacks = y_test[attack_mask]
    sample_size = min(50, len(X_attacks))
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X_attacks), sample_size, replace=False)
    X_sample = X_attacks[sample_idx]
    y_sample = y_attacks[sample_idx]

    print(f"\nUsing {sample_size} malicious samples for sweep")

    # Train surrogate for white-box attacks
    print("Training surrogate model...")
    surrogate, _ = train_surrogate(X_train, y_train)

    # Zero-Trust engine for bypass rate
    zt_engine = ZeroTrustEngine()

    results = []

    for eps in EPS_VALUES:
        print(f"\n{'─'*50}")
        print(f"  Epsilon = {eps}")
        print(f"{'─'*50}")

        # --- FGSM Attack ---
        try:
            X_adv_fgsm = run_whitebox_attack(
                surrogate, rf, X_sample, y_sample,
                clip_values, sample_size=sample_size, eps=eps
            )
            if isinstance(X_adv_fgsm, tuple):
                X_adv_fgsm = X_adv_fgsm[0] if len(X_adv_fgsm) > 0 else X_sample

            # Validate constraints
            constraint_result = validate_adversarial_batch(X_adv_fgsm)

            # Evasion: model predicts benign for adversarial
            fgsm_preds = rf.predict(X_adv_fgsm)
            evasion_rate_fgsm = float(np.mean(fgsm_preds == 0))  # Predicted benign = evasion
            robust_acc_fgsm = float(np.mean(fgsm_preds == y_sample))

            # Perturbation norms
            l2_fgsm, linf_fgsm = compute_perturbation_norms(X_sample, X_adv_fgsm)

            # ZT bypass rate
            risk_scores = rf.predict_proba(X_adv_fgsm)[:, 1]
            contexts = generate_attacker_context(len(X_adv_fgsm), seed=42)
            zt_decisions = zt_engine.evaluate_batch(risk_scores, contexts)
            zt_bypass_fgsm = float(np.mean([d.decision == "ALLOW" for d in zt_decisions]))

            results.append({
                'attack': 'FGSM',
                'epsilon': eps,
                'evasion_rate': evasion_rate_fgsm,
                'robust_accuracy': robust_acc_fgsm,
                'avg_l2': l2_fgsm,
                'avg_linf': linf_fgsm,
                'zt_bypass_rate': zt_bypass_fgsm,
                'constraint_pass_rate': constraint_result['pass_rate'],
            })

            print(f"  FGSM: evasion={evasion_rate_fgsm:.1%}, "
                  f"robust_acc={robust_acc_fgsm:.1%}, "
                  f"ZT_bypass={zt_bypass_fgsm:.1%}")
        except Exception as e:
            print(f"  FGSM failed at eps={eps}: {e}")
            results.append({
                'attack': 'FGSM', 'epsilon': eps,
                'evasion_rate': 0, 'robust_accuracy': 1, 'avg_l2': 0,
                'avg_linf': 0, 'zt_bypass_rate': 0, 'constraint_pass_rate': 0,
                'error': str(e),
            })

        # --- PGD Attack (using FGSM with multiple steps as approximation) ---
        try:
            alpha = eps / PGD_ALPHA_FACTOR
            X_adv_pgd = X_sample.copy().astype(float)

            surrogate.eval()
            y_tensor = torch.tensor(y_sample, dtype=torch.long)
            criterion = nn.CrossEntropyLoss()

            for restart in range(PGD_RESTARTS):
                # Random init
                X_pgd_current = X_sample + np.random.uniform(
                    -eps, eps, X_sample.shape
                )
                X_pgd_current = np.clip(X_pgd_current, feature_min, feature_max)

                for step in range(PGD_ITERATIONS):
                    try:
                        X_pgd_tensor = torch.tensor(X_pgd_current, dtype=torch.float32, requires_grad=True)
                        outputs = surrogate(X_pgd_tensor)
                        loss = criterion(outputs, y_tensor)
                        
                        surrogate.zero_grad()
                        loss.backward()
                        
                        grad = X_pgd_tensor.grad.detach().numpy()
                        
                        delta = X_pgd_current + alpha * np.sign(grad) - X_sample
                        # Project back to eps-ball
                        delta = np.clip(delta, -eps, eps)
                        X_pgd_current = np.clip(X_sample + delta, feature_min, feature_max)
                    except Exception as e:
                        print(f"PGD step failed: {e}")
                        break

                X_adv_pgd = X_pgd_current

            pgd_preds = rf.predict(X_adv_pgd)
            evasion_rate_pgd = float(np.mean(pgd_preds == 0))
            robust_acc_pgd = float(np.mean(pgd_preds == y_sample))
            l2_pgd, linf_pgd = compute_perturbation_norms(X_sample, X_adv_pgd)

            risk_scores_pgd = rf.predict_proba(X_adv_pgd)[:, 1]
            contexts_pgd = generate_attacker_context(len(X_adv_pgd), seed=43)
            zt_decisions_pgd = zt_engine.evaluate_batch(risk_scores_pgd, contexts_pgd)
            zt_bypass_pgd = float(np.mean([d.decision == "ALLOW" for d in zt_decisions_pgd]))

            constraint_pgd = validate_adversarial_batch(X_adv_pgd)

            results.append({
                'attack': 'PGD',
                'epsilon': eps,
                'evasion_rate': evasion_rate_pgd,
                'robust_accuracy': robust_acc_pgd,
                'avg_l2': l2_pgd,
                'avg_linf': linf_pgd,
                'zt_bypass_rate': zt_bypass_pgd,
                'constraint_pass_rate': constraint_pgd['pass_rate'],
                'pgd_iterations': PGD_ITERATIONS,
                'pgd_alpha': alpha,
                'pgd_restarts': PGD_RESTARTS,
            })

            print(f"  PGD:  evasion={evasion_rate_pgd:.1%}, "
                  f"robust_acc={robust_acc_pgd:.1%}, "
                  f"ZT_bypass={zt_bypass_pgd:.1%}")
        except Exception as e:
            print(f"  PGD failed at eps={eps}: {e}")
            results.append({
                'attack': 'PGD', 'epsilon': eps,
                'evasion_rate': 0, 'robust_accuracy': 1, 'avg_l2': 0,
                'avg_linf': 0, 'zt_bypass_rate': 0, 'constraint_pass_rate': 0,
                'error': str(e),
            })

    # Save results
    results_path = os.path.join(RESULTS_DIR, "epsilon_sweep.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Generate Figure 3: Dual-axis epsilon sweep chart
    try:
        df = pd.DataFrame(results)
        fig, ax1 = plt.subplots(figsize=(10, 6))

        for attack in ['FGSM', 'PGD']:
            mask = df['attack'] == attack
            ax1.plot(df[mask]['epsilon'], df[mask]['evasion_rate'],
                     'o-', label=f'{attack} Evasion Rate')

        ax1.set_xlabel('Epsilon (Perturbation Magnitude)')
        ax1.set_ylabel('Evasion Success Rate')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        for attack in ['FGSM', 'PGD']:
            mask = df['attack'] == attack
            ax2.plot(df[mask]['epsilon'], df[mask]['zt_bypass_rate'],
                     's--', label=f'{attack} ZT Bypass Rate')

        ax2.set_ylabel('Zero-Trust Bypass Rate')
        ax2.legend(loc='upper right')

        plt.title('Adversarial Evasion vs Zero-Trust Bypass Rate')
        plt.tight_layout()
        fig_path = os.path.join(FIGURES_DIR, "epsilon_sweep.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nFigure saved to: {fig_path}")
    except Exception as e:
        print(f"Figure generation failed: {e}")

    # Print summary
    print("\n" + "=" * 80)
    print(f"  {'Attack':<6s} {'Eps':>5s} {'Evasion%':>9s} {'RobAcc%':>8s} "
          f"{'L2':>6s} {'L∞':>6s} {'ZT-Bypass%':>10s} {'Valid%':>7s}")
    print("=" * 80)
    for r in results:
        if 'error' not in r:
            print(f"  {r['attack']:<6s} {r['epsilon']:>5.2f} "
                  f"{r['evasion_rate']:>8.1%} {r['robust_accuracy']:>7.1%} "
                  f"{r['avg_l2']:>6.3f} {r['avg_linf']:>6.3f} "
                  f"{r['zt_bypass_rate']:>9.1%} {r['constraint_pass_rate']:>6.0%}")
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
