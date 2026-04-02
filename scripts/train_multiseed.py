"""
Multi-seed training script for statistically rigorous evaluation.

Trains the Random Forest classifier across 5 independent random seeds
and records accuracy, precision, recall, F1-score, and ROC-AUC for each.
Results are saved as mean ± standard deviation with 95% confidence
intervals, as required for publication.

Usage:
    python scripts/train_multiseed.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
import joblib

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import (
    DATA_DIR, MODEL_DIR, RESULTS_DIR,
    MULTI_SEED_VALUES, ISOLATION_CONTAMINATION
)
from src.evaluation.statistics import multi_seed_aggregate


def train_single_seed(X_train, y_train, X_test, y_test, seed):
    """Train and evaluate a Random Forest model with a given seed.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        seed: Random seed for this run.

    Returns:
        Dict with accuracy, precision, recall, f1, roc_auc, and the model.
    """
    print(f"\n{'='*50}")
    print(f"  Training with seed={seed}")
    print(f"{'='*50}")

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=seed)
    rf.fit(X_train, y_train)

    # Train Isolation Forest on benign data only
    X_train_benign = X_train[y_train == 0]
    iso = IsolationForest(
        n_estimators=100,
        contamination=ISOLATION_CONTAMINATION,
        random_state=seed
    )
    iso.fit(X_train_benign)

    # Evaluate
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    metrics = {
        'seed': seed,
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_prob)),
    }

    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

    return metrics, rf, iso



def main():
    """Run multi-seed training and evaluation pipeline."""
    print("=" * 60)
    print("  MULTI-SEED TRAINING PIPELINE")
    print(f"  Seeds: {MULTI_SEED_VALUES}")
    print("=" * 60)

    # Load dataset
    train_df = pd.read_csv(os.path.join(MODEL_DIR, "train_set.csv"))
    test_df = pd.read_csv(os.path.join(MODEL_DIR, "test_set.csv"))

    X_train = train_df.drop(columns=['label']).values
    y_train = train_df['label'].values
    X_test = test_df.drop(columns=['label']).values
    y_test = test_df['label'].values

    print(f"\nTrain Dataset: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test Dataset: {X_test.shape[0]} samples, {X_test.shape[1]} features")

    # Run training across all seeds
    all_metrics = []
    for seed in MULTI_SEED_VALUES:
        metrics, rf, iso = train_single_seed(
            X_train, y_train, X_test, y_test, seed
        )
        all_metrics.append(metrics)

        # Save per-seed model
        seed_dir = os.path.join(MODEL_DIR, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        joblib.dump(rf, os.path.join(seed_dir, "random_forest.pkl"))
        joblib.dump(iso, os.path.join(seed_dir, "isolation_forest.pkl"))

    # Compute aggregate statistics
    agg = multi_seed_aggregate(all_metrics)

    # Build results payload
    results = {
        'experiment': 'multi_seed_training',
        'seeds': MULTI_SEED_VALUES,
        'n_seeds': len(MULTI_SEED_VALUES),
        'dataset': 'combined_traffic (NSL-KDD derived)',
        'train_size': len(X_train),
        'test_size': len(X_test),
        'per_seed_results': all_metrics,
        'aggregate': {k: {kk: vv for kk, vv in v.items() if kk != 'per_seed'}
                      for k, v in agg.items()},
    }

    # Save results
    results_path = os.path.join(RESULTS_DIR, "multiseed_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("  AGGREGATE RESULTS (mean ± std, 95% CI)")
    print("=" * 60)
    for name, stats in agg.items():
        print(f"  {name:12s}: {stats['formatted']}")

    print(f"\nResults saved to: {results_path}")
    print(f"Per-seed models saved to: {MODEL_DIR}/seed_*/")


if __name__ == "__main__":
    main()
