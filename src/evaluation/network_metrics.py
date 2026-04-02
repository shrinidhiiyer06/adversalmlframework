from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_network_performance(y_true, y_pred, y_scores):
    """
    Comprehensive performance evaluation with security-focused metrics.
    
    Args:
        y_true: True labels (0=benign, 1=malicious)
        y_pred: Predicted labels (binary)
        y_scores: Risk scores/probabilities (0-1)
        
    Returns:
        results: Dictionary with metrics and artifacts for plotting
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    # Handle single-class cases (common in adversarial evaluation)
    n_classes = len(np.unique(y_true))
    
    if n_classes > 1:
        report = classification_report(y_true, y_pred, labels=[0, 1], output_dict=True, zero_division=0)
        fpr_list, tpr_list, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr_list, tpr_list)
    else:
        # Fallback for single-class (e.g., only malicious samples)
        report = classification_report(y_true, y_pred, labels=[0, 1], output_dict=True, zero_division=0)
        fpr_list, tpr_list = np.array([0, 1]), np.array([0, 1])
        roc_auc = 0.0
    
    # SOC/Security Metrics
    # FPR: Probability of false alarm
    # FNR: Probability of missing an attack (Detection Gap)
    false_positive_rate = cm[0][1] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
    false_negative_rate = cm[1][0] / (cm[1][0] + cm[1][1]) if (cm[1][0] + cm[1][1]) > 0 else 0
    
    return {
        "confusion_matrix": cm,
        "classification_report": report,
        "roc_auc": roc_auc,
        "fpr_list": fpr_list,
        "tpr_list": tpr_list,
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
        "precision": float(report['1']['precision']),
        "recall": float(report['1']['recall']),
        "f1_score": float(report['1']['f1-score'])
    }

def plot_confusion_matrix(cm, labels=['Normal', 'Attack']):
    """Returns a matplotlib figure of the confusion matrix"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Network Intrusion Confusion Matrix")
    return fig

def plot_roc_curve(fpr, tpr, roc_auc):
    """Returns a matplotlib figure of the ROC curve"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (area = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.set_title('Receiver Operating Characteristic (ROC)')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    return fig
