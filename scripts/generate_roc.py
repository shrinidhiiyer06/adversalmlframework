"""
ROC curve and threshold analysis script.

Generates:
  - ROC curve saved as figures/roc_curve.png (Figure 2)
  - Threshold sweep at [0.3, 0.4, 0.5, 0.6, 0.7] showing precision, recall, F1
  - Confusion matrix saved as figures/confusion_matrix.png (Figure 5)

Demonstrates that the model's operating point is a deliberate choice
to maximize precision rather than an artifact of poor training.

Usage:
    python scripts/generate_roc.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import MODEL_DIR, RESULTS_DIR, FIGURES_DIR


def main():
    """Generate ROC curve, threshold sweep, and confusion matrix."""
    print("=" * 60)
    print("  ROC CURVE AND THRESHOLD ANALYSIS")
    print("=" * 60)

    # Load model and test data
    rf = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
    test_df = pd.read_csv(os.path.join(MODEL_DIR, "test_set.csv"))
    X_test = test_df.drop(columns=['label']).values
    y_test = test_df['label'].values

    # Get probabilities
    y_prob = rf.predict_proba(X_test)[:, 1]

    # ---- ROC Curve (Figure 2) ----
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='#636EFA', lw=2,
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--',
            label='Random classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Receiver Operating Characteristic — Baseline Classifier', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    roc_path = os.path.join(FIGURES_DIR, "roc_curve.png")
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nROC curve saved to: {roc_path}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    # ---- Threshold Sweep ----
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    sweep_results = []

    print(f"\n{'Threshold':>10s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s}")
    print("-" * 45)

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        sweep_results.append({
            'threshold': thresh,
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
        })
        print(f"{thresh:>10.1f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f}")

    # ---- Confusion Matrix (Figure 5) ----
    y_pred_default = rf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_default)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=['Benign', 'Attack'])
    disp.plot(ax=ax, cmap='Blues', colorbar=True)
    ax.set_title('Confusion Matrix — Baseline Classifier on NSL-KDD Test Set',
                 fontsize=13)

    cm_path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nConfusion matrix saved to: {cm_path}")

    # ---- Save all results ----
    results = {
        'experiment': 'roc_threshold_analysis',
        'roc_auc': float(roc_auc),
        'threshold_sweep': sweep_results,
        'confusion_matrix': {
            'tn': int(cm[0][0]),
            'fp': int(cm[0][1]),
            'fn': int(cm[1][0]),
            'tp': int(cm[1][1]),
        },
        'default_threshold': 0.5,
    }

    results_path = os.path.join(RESULTS_DIR, "roc_threshold.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
