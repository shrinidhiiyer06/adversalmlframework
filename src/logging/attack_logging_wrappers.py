"""
Attack Logging Wrappers

Wrapper functions that add logging capabilities to existing attack modules.
Drop-in replacements for run_blackbox_attack, run_whitebox_attack, etc.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from .log_manager import LogManager


def run_blackbox_attack_with_logging(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    clip_values: Tuple[float, float],
    log_manager: Optional[LogManager] = None,
    sample_size: int = 100,
    max_iter: int = 50,
    max_eval: int = 100,
    init_eval: int = 10,
    random_state: Optional[int] = None,
    verbose: bool = False,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    """
    Run HopSkipJump attack with automatic logging.
    
    This is a drop-in replacement for the original run_blackbox_attack
    that adds comprehensive logging capabilities.
    
    Args:
        Same as original run_blackbox_attack, plus:
        log_manager: LogManager instance (creates one if None)
        
    Returns:
        Same as original (X_adv, X_sample, y_sample, avg_queries, total_queries)
    """
    # Import the actual attack function
    try:
        from src.attacks.blackbox import run_blackbox_attack
    except ImportError:
        raise ImportError("Cannot import run_blackbox_attack. Ensure src.attacks.blackbox is available.")
    
    # Create log manager if not provided
    if log_manager is None:
        log_manager = LogManager()
    
    # Run the actual attack
    X_adv, X_sample, y_sample, avg_queries, total_queries = run_blackbox_attack(
        model=model,
        X_test=X_test,
        y_test=y_test,
        clip_values=clip_values,
        sample_size=sample_size,
        max_iter=max_iter,
        max_eval=max_eval,
        init_eval=init_eval,
        random_state=random_state,
        verbose=verbose
    )
    
    # Get model predictions for logging
    preds_orig = model.predict(X_sample)
    preds_adv = model.predict(X_adv)
    
    # Get confidence scores if available
    if hasattr(model, 'predict_proba'):
        probs_adv = model.predict_proba(X_adv)
        confidences = np.max(probs_adv, axis=1)
    else:
        confidences = np.ones(len(X_adv))
    
    # Calculate attack success
    attack_success = (preds_orig != preds_adv)
    success_rate = np.mean(attack_success)
    
    # Calculate perturbation norms
    l2_norms = np.linalg.norm(X_adv - X_sample, ord=2, axis=1)
    linf_norms = np.linalg.norm(X_adv - X_sample, ord=np.inf, axis=1)
    
    # Log batch attack summary
    log_manager.log_batch_attack(
        attack_type="HopSkipJump",
        summary_stats={
            "total_samples": sample_size,
            "successful_attacks": int(np.sum(attack_success)),
            "success_rate": float(success_rate),
            "avg_queries_per_sample": float(avg_queries),
            "total_queries": int(total_queries),
            "max_iterations": max_iter,
            "max_eval": max_eval,
            "mean_l2_perturbation": float(np.mean(l2_norms)),
            "mean_linf_perturbation": float(np.mean(linf_norms)),
            "median_confidence": float(np.median(confidences)),
            "random_state": random_state
        },
        individual_results=[
            {
                "sample_id": i,
                "success": bool(attack_success[i]),
                "original_pred": int(preds_orig[i]),
                "adversarial_pred": int(preds_adv[i]),
                "true_label": int(y_sample[i]),
                "confidence": float(confidences[i]),
                "l2_norm": float(l2_norms[i]),
                "linf_norm": float(linf_norms[i])
            }
            for i in range(min(len(X_adv), 100))  # Log first 100 samples
        ],
        metadata={
            "attack_category": "black-box",
            "query_based": True,
            "gradient_free": True,
            "clip_values": list(clip_values) if isinstance(clip_values, tuple) else clip_values
        }
    )
    
    return X_adv, X_sample, y_sample, avg_queries, total_queries


def run_whitebox_attack_with_logging(
    surrogate_model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    clip_values: Tuple[float, float],
    log_manager: Optional[LogManager] = None,
    sample_size: int = 100,
    eps: Optional[float] = None,
    random_state: Optional[int] = None,
    norm: str = 'inf',
    minimal: bool = False,
    batch_size: int = 128,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run FGM attack with automatic logging.
    
    Drop-in replacement for run_whitebox_attack with logging.
    
    Args:
        Same as original run_whitebox_attack, plus:
        log_manager: LogManager instance (creates one if None)
        
    Returns:
        Same as original (X_adv, X_sample, y_sample)
    """
    # Import the actual attack function
    try:
        from src.attacks.whitebox import run_whitebox_attack
    except ImportError:
        raise ImportError("Cannot import run_whitebox_attack. Ensure src.attacks.whitebox is available.")
    
    # Create log manager if not provided
    if log_manager is None:
        log_manager = LogManager()
    
    # Run the actual attack
    X_adv, X_sample, y_sample = run_whitebox_attack(
        surrogate_model=surrogate_model,
        X_test=X_test,
        y_test=y_test,
        clip_values=clip_values,
        sample_size=sample_size,
        eps=eps,
        random_state=random_state,
        norm=norm,
        minimal=minimal,
        batch_size=batch_size
    )
    
    # Calculate perturbation statistics
    l2_norms = np.linalg.norm(X_adv - X_sample, ord=2, axis=1)
    linf_norms = np.linalg.norm(X_adv - X_sample, ord=np.inf, axis=1)
    
    # Log batch attack
    log_manager.log_batch_attack(
        attack_type="FastGradientMethod",
        summary_stats={
            "total_samples": sample_size,
            "epsilon": float(eps) if eps is not None else 0.1,
            "norm": norm,
            "minimal": minimal,
            "batch_size": batch_size,
            "mean_l2_perturbation": float(np.mean(l2_norms)),
            "mean_linf_perturbation": float(np.mean(linf_norms)),
            "max_l2_perturbation": float(np.max(l2_norms)),
            "max_linf_perturbation": float(np.max(linf_norms)),
            "random_state": random_state
        },
        individual_results=[
            {
                "sample_id": i,
                "true_label": int(y_sample[i]),
                "l2_norm": float(l2_norms[i]),
                "linf_norm": float(linf_norms[i])
            }
            for i in range(min(len(X_adv), 100))
        ],
        metadata={
            "attack_category": "white-box",
            "gradient_based": True,
            "surrogate_attack": True,
            "clip_values": list(clip_values) if isinstance(clip_values, tuple) else clip_values
        }
    )
    
    return X_adv, X_sample, y_sample


def ensemble_defense_predict_with_logging(
    rf,
    iso_forest,
    X: np.ndarray,
    log_manager: Optional[LogManager] = None,
    conf_threshold: float = 0.15,
    return_latency: bool = False,
    defense_mode: str = "Standard",
    **kwargs
) -> np.ndarray:
    """
    Run ensemble defense with automatic logging of decisions.
    
    Drop-in replacement for ensemble_defense_predict with logging.
    
    Args:
        Same as original ensemble_defense_predict, plus:
        log_manager: LogManager instance (creates one if None)
        defense_mode: Defense mode string for logging
        
    Returns:
        Same as original (predictions or (predictions, latency))
    """
    # Import the actual defense function
    try:
        from src.core.defense import ensemble_defense_predict, get_defense_explanation
    except ImportError:
        raise ImportError("Cannot import defense functions. Ensure src.core.defense is available.")
    
    # Create log manager if not provided
    if log_manager is None:
        log_manager = LogManager()
    
    # Run the actual defense with explanation
    result = ensemble_defense_predict(
        rf=rf,
        iso_forest=iso_forest,
        X=X,
        conf_threshold=conf_threshold,
        return_latency=True  # Always get latency for logging
    )
    
    # Unpack result
    if isinstance(result, tuple):
        predictions, latency_ms = result
    else:
        predictions = result
        latency_ms = 0.0
    
    # Get detailed explanation
    explanation = get_defense_explanation(
        rf=rf,
        iso_forest=iso_forest,
        X=X,
        conf_threshold=conf_threshold
    )
    
    # Log defense events (sample first 10 for performance)
    num_to_log = min(10, len(X))
    for i in range(num_to_log):
        log_manager.log_defense_event(
            defense_mode=defense_mode,
            input_sample={
                "sample_index": i,
                "feature_vector_size": X.shape[1],
                "feature_stats": {
                    "mean": float(np.mean(X[i])),
                    "std": float(np.std(X[i])),
                    "min": float(np.min(X[i])),
                    "max": float(np.max(X[i]))
                }
            },
            decision={
                "action": "DENY" if predictions[i] == 1 else "ALLOW",
                "prediction": int(predictions[i]),
                "confidence": float(explanation['rf_probabilities'][i]),
                "latency_ms": float(latency_ms / len(X))  # Per-sample latency
            },
            detection_flags={
                "rf_attack": bool(explanation['rf_predictions'][i] == 1),
                "anomaly": bool(explanation['anomaly_flags'][i]),
                "uncertain": bool(explanation['uncertainty_flags'][i])
            },
            metadata={
                "confidence_threshold": conf_threshold,
                "deny_reason": str(explanation['deny_reasons'][i])
            }
        )
    
    # Return based on original function's expected return
    if return_latency:
        return predictions, latency_ms
    else:
        return predictions


# Convenience function to create a logging-enabled session
def create_logged_attack_session(base_dir: str = "logs") -> LogManager:
    """
    Create a new logging session for attacks and defenses.
    
    Args:
        base_dir: Base directory for logs
        
    Returns:
        LogManager instance ready to use
        
    Example:
        >>> log_mgr = create_logged_attack_session()
        >>> X_adv, X_orig, y_orig, avg_q, total_q = run_blackbox_attack_with_logging(
        ...     model, X_test, y_test, clip_values,
        ...     log_manager=log_mgr,
        ...     sample_size=100
        ... )
        >>> log_mgr.export_logs(format='json')
    """
    return LogManager(base_dir=base_dir)


def export_session_logs_all_formats(
    log_manager: LogManager,
    base_filename: str = "attack_session"
) -> Dict[str, str]:
    """
    Export logs in all supported formats.
    
    Args:
        log_manager: LogManager instance with logged events
        base_filename: Base name for exported files
        
    Returns:
        Dictionary mapping format to filepath
    """
    exports = {}
    
    for fmt in ['json', 'txt', 'md', 'csv']:
        filepath = log_manager.export_logs(format=fmt, filename=base_filename)
        exports[fmt] = filepath
    
    return exports
