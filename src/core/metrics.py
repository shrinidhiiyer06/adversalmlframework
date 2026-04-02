"""
Evaluation metrics for adversarial attack assessment.

Provides standard metrics for measuring attack effectiveness and model robustness.
All functions include input validation and handle edge cases properly.
"""

import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def validate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Validate prediction arrays.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(y_true, np.ndarray):
        raise TypeError(f"y_true must be numpy array, got {type(y_true)}")
    
    if not isinstance(y_pred, np.ndarray):
        raise TypeError(f"y_pred must be numpy array, got {type(y_pred)}")
    
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} samples, "
            f"y_pred has {len(y_pred)} samples"
        )
    
    if len(y_true) == 0:
        raise ValueError("Cannot compute metrics on empty arrays")


def calculate_evasion_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate evasion rate: percentage of attacks misclassified as benign.
    
    Evasion Rate = (# of attacks predicted as benign) / (# of attacks)
    
    A successful attack evades detection by being misclassified as benign.
    Higher evasion rate = more successful attack = worse for defender.
    
    Args:
        y_true: True labels where 1 = attack, 0 = benign
        y_pred: Predicted labels where 1 = attack, 0 = benign
        
    Returns:
        Evasion rate in range [0.0, 1.0]
        Returns 0.0 if there are no attacks in y_true
        
    Example:
        >>> y_true = np.array([1, 1, 1, 0, 0])  # 3 attacks, 2 benign
        >>> y_pred = np.array([0, 1, 0, 0, 0])  # 2 attacks evaded detection
        >>> evasion_rate = calculate_evasion_rate(y_true, y_pred)
        >>> print(f"Evasion rate: {evasion_rate:.2%}")
        Evasion rate: 66.67%
    """
    # Validate inputs
    validate_predictions(y_true, y_pred)
    
    # Find indices where true label is attack (1)
    attack_indices = np.where(y_true == 1)[0]
    
    # If no attacks in the dataset, return 0.0
    if len(attack_indices) == 0:
        logger.warning("No attacks found in y_true, returning evasion_rate=0.0")
        return 0.0
    
    # Count how many attacks were predicted as benign (0)
    # These are the "evasions" - attacks that fooled the detector
    evasions = np.sum(y_pred[attack_indices] == 0)
    
    # Calculate evasion rate
    evasion_rate = float(evasions) / len(attack_indices)
    
    return evasion_rate


def calculate_detection_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate detection rate: percentage of attacks correctly identified.
    
    Detection Rate = 1 - Evasion Rate
                   = (# of attacks correctly detected) / (# of attacks)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Detection rate in range [0.0, 1.0]
    """
    return 1.0 - calculate_evasion_rate(y_true, y_pred)


def calculate_false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate false positive rate: percentage of benign traffic blocked.
    
    FPR = (# of benign predicted as attacks) / (# of benign)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        False positive rate in range [0.0, 1.0]
        Returns 0.0 if there are no benign samples in y_true
    """
    validate_predictions(y_true, y_pred)
    
    # Find indices where true label is benign (0)
    benign_indices = np.where(y_true == 0)[0]
    
    if len(benign_indices) == 0:
        logger.warning("No benign samples found in y_true, returning FPR=0.0")
        return 0.0
    
    # Count how many benign samples were predicted as attacks (1)
    false_positives = np.sum(y_pred[benign_indices] == 1)
    
    fpr = float(false_positives) / len(benign_indices)
    
    return fpr


def get_perturbation_norms(X_orig: np.ndarray, X_adv: np.ndarray) -> Tuple[float, float]:
    """
    Calculate L2 and L-infinity norms of perturbations.
    
    Perturbation norms measure how much the adversarial examples differ
    from the original samples. Lower norms = more subtle attack.
    
    Args:
        X_orig: Original samples of shape (n_samples, n_features)
        X_adv: Adversarial samples of shape (n_samples, n_features)
        
    Returns:
        Tuple of (mean_l2_norm, mean_linf_norm):
            - mean_l2_norm: Average L2 (Euclidean) distance
            - mean_linf_norm: Average L-infinity (max element-wise) distance
            
    Example:
        >>> X_orig = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> X_adv = np.array([[1.1, 2.0], [3.0, 4.1]])
        >>> l2, linf = get_perturbation_norms(X_orig, X_adv)
        >>> print(f"L2: {l2:.4f}, L-inf: {linf:.4f}")
        L2: 0.1000, L-inf: 0.1000
        
    Notes:
        - L2 norm: sqrt(sum of squared differences) per sample
        - L-infinity norm: maximum absolute difference per sample
        - Both are averaged across all samples
    """
    # Validate inputs
    if not isinstance(X_orig, np.ndarray) or not isinstance(X_adv, np.ndarray):
        raise TypeError("X_orig and X_adv must be numpy arrays")
    
    if X_orig.shape != X_adv.shape:
        raise ValueError(
            f"Shape mismatch: X_orig has shape {X_orig.shape}, "
            f"X_adv has shape {X_adv.shape}"
        )
    
    if len(X_orig) == 0:
        raise ValueError("Cannot compute perturbation norms on empty arrays")
    
    if X_orig.ndim != 2:
        raise ValueError(f"Arrays must be 2D, got {X_orig.ndim}D")
    
    try:
        # Compute perturbations (differences)
        perturbations = X_adv - X_orig
        
        # L2 norm: sqrt(sum of squared differences) for each sample
        l2_norms = np.linalg.norm(perturbations, ord=2, axis=1)
        mean_l2 = float(np.mean(l2_norms))
        
        # L-infinity norm: maximum absolute difference for each sample
        linf_norms = np.linalg.norm(perturbations, ord=np.inf, axis=1)
        mean_linf = float(np.mean(linf_norms))
        
        # Validate results
        if not np.isfinite(mean_l2) or not np.isfinite(mean_linf):
            raise ValueError("Computed norms contain non-finite values")
        
        return mean_l2, mean_linf
        
    except Exception as e:
        logger.error(f"Perturbation norm calculation failed: {e}")
        raise RuntimeError(f"Cannot compute perturbation norms: {e}") from e


def calculate_attack_success_rate(
    y_true: np.ndarray,
    y_pred_orig: np.ndarray,
    y_pred_adv: np.ndarray
) -> float:
    """
    Calculate attack success rate: percentage of samples where prediction changed.
    
    ASR = (# of samples where pred_orig != pred_adv) / (# of samples)
    
    This measures how often the adversarial perturbation successfully
    changed the model's prediction (regardless of whether it's correct).
    
    Args:
        y_true: True labels
        y_pred_orig: Predictions on original samples
        y_pred_adv: Predictions on adversarial samples
        
    Returns:
        Attack success rate in range [0.0, 1.0]
    """
    validate_predictions(y_true, y_pred_orig)
    validate_predictions(y_true, y_pred_adv)
    
    # Count how many predictions changed
    changed = np.sum(y_pred_orig != y_pred_adv)
    
    asr = float(changed) / len(y_true)
    
    return asr


def calculate_robustness_score(
    accuracy_clean: float,
    accuracy_adv: float
) -> float:
    """
    Calculate robustness score: how much accuracy degrades under attack.
    
    Robustness = accuracy_adv / accuracy_clean
    
    A robustness score of 1.0 means no degradation (perfect robustness).
    A robustness score of 0.5 means accuracy dropped by 50%.
    
    Args:
        accuracy_clean: Accuracy on clean data
        accuracy_adv: Accuracy on adversarial data
        
    Returns:
        Robustness score in range [0.0, 1.0]
    """
    if not 0 <= accuracy_clean <= 1:
        raise ValueError(f"accuracy_clean must be in [0, 1], got {accuracy_clean}")
    
    if not 0 <= accuracy_adv <= 1:
        raise ValueError(f"accuracy_adv must be in [0, 1], got {accuracy_adv}")
    
    if accuracy_clean == 0:
        logger.warning("accuracy_clean is 0, returning robustness=0.0")
        return 0.0
    
    robustness = accuracy_adv / accuracy_clean
    
    return float(robustness)


def get_confusion_matrix_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate all confusion matrix-based metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with metrics:
            - true_positives: Attacks correctly identified
            - true_negatives: Benign correctly identified
            - false_positives: Benign incorrectly flagged as attacks
            - false_negatives: Attacks that evaded detection
            - precision: TP / (TP + FP)
            - recall: TP / (TP + FN) [same as detection rate]
            - f1_score: Harmonic mean of precision and recall
            - evasion_rate: FN / (TP + FN)
            - fpr: FP / (TN + FP)
    """
    validate_predictions(y_true, y_pred)
    
    # Calculate confusion matrix components
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Derived metrics with zero-division handling
    precision = float(tp) / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    evasion_rate = float(fn) / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = float(fp) / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'evasion_rate': evasion_rate,
        'false_positive_rate': fpr
    }
