"""
Epsilon sweep module for robustness curve generation.

This module performs systematic evaluation of model robustness across
different perturbation magnitudes, comparing baseline and defended models.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Union
import torch.nn as nn
import torch.optim as optim
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from sklearn.metrics import accuracy_score
import logging

# Note: These imports should be adjusted to match your project structure
# from src.core.defense import ensemble_defense_predict
# from src.config import EPS_VALUES

logger = logging.getLogger(__name__)


def validate_sweep_inputs(
    X_test: np.ndarray,
    y_test: np.ndarray,
    clip_values: Tuple[float, float],
    sample_size: int,
    eps_values: List[float]
) -> None:
    """
    Validate inputs for epsilon sweep.
    
    Args:
        X_test: Test features
        y_test: Test labels
        clip_values: Min/max values for clipping
        sample_size: Number of samples to use
        eps_values: List of epsilon values to sweep
        
    Raises:
        ValueError: If validation fails
    """
    if len(X_test) != len(y_test):
        raise ValueError(f"X_test and y_test length mismatch: {len(X_test)} != {len(y_test)}")
    
    if sample_size > len(X_test):
        raise ValueError(f"sample_size ({sample_size}) exceeds dataset size ({len(X_test)})")
    
    if sample_size < 1:
        raise ValueError(f"sample_size must be positive, got {sample_size}")
    
    if not isinstance(clip_values, (tuple, list)) or len(clip_values) != 2:
        raise ValueError(f"clip_values must be a tuple of (min, max), got {clip_values}")
    
    if np.any(clip_values[0] >= clip_values[1]):
        raise ValueError(f"clip_values[0] must be < clip_values[1], got {clip_values}")
    
    if not eps_values or len(eps_values) == 0:
        raise ValueError("eps_values cannot be empty")
    
    if any(eps < 0 for eps in eps_values):
        raise ValueError(f"All epsilon values must be non-negative, got {eps_values}")
    
    if not np.isfinite(X_test).all():
        raise ValueError("X_test contains non-finite values (NaN or inf)")


def safe_ensemble_predict(
    rf_model,
    iso_forest,
    X: np.ndarray,
    ensemble_defense_predict_func
) -> np.ndarray:
    """
    Safely call ensemble_defense_predict with consistent return handling.
    
    This function handles the case where ensemble_defense_predict may return
    either a tuple (predictions, metadata) or just predictions.
    
    Args:
        rf_model: Random Forest model
        iso_forest: Isolation Forest model
        X: Input features
        ensemble_defense_predict_func: The ensemble defense function
        
    Returns:
        Predictions array of shape (n_samples,)
        
    Raises:
        ValueError: If predictions have unexpected shape or type
    """
    try:
        result = ensemble_defense_predict_func(rf_model, iso_forest, X)
        
        # Handle tuple return (predictions, metadata)
        if isinstance(result, tuple):
            predictions = result[0]
            logger.debug(f"Ensemble returned tuple with {len(result)} elements")
        else:
            predictions = result
        
        # Validate predictions shape
        predictions = np.asarray(predictions)
        if predictions.ndim > 1:
            predictions = predictions.flatten()
        
        if len(predictions) != len(X):
            raise ValueError(
                f"Prediction length mismatch: expected {len(X)}, got {len(predictions)}"
            )
        
        return predictions
        
    except Exception as e:
        logger.error(f"Ensemble prediction failed: {str(e)}")
        logger.error(f"Input shape: {X.shape}, Result type: {type(result)}")
        raise ValueError(f"Ensemble defense prediction failed: {str(e)}") from e


def run_epsilon_sweep(
    rf_model,
    iso_forest,
    surrogate_model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    clip_values: Tuple[float, float],
    eps_values: List[float],
    ensemble_defense_predict_func,
    sample_size: int = 50,
    random_state: int = 42,
    enable_debug_logging: bool = False
) -> pd.DataFrame:
    """
    Generate robustness curves for baseline and defended models.
    
    Sweeps across epsilon values to measure accuracy degradation under
    adversarial attack, comparing baseline model performance against
    the ensemble defense mechanism.
    
    Args:
        rf_model: Trained Random Forest classifier
        iso_forest: Trained Isolation Forest for anomaly detection
        surrogate_model: PyTorch surrogate model for generating attacks
        X_test: Test features of shape (n_samples, n_features)
        y_test: Test labels of shape (n_samples,)
        clip_values: Tuple of (min, max) for clipping adversarial examples
        eps_values: List of epsilon values to sweep (perturbation magnitudes)
        ensemble_defense_predict_func: Function to call ensemble defense
        sample_size: Number of samples to use (default: 50)
        random_state: Random seed for reproducibility (default: 42)
        enable_debug_logging: Whether to enable detailed debug logs (default: False)
        
    Returns:
        DataFrame with columns: ['epsilon', 'Baseline', 'ZT-Shield (Defended)']
        Each row contains accuracy for a specific epsilon value.
        
    Raises:
        ValueError: If input validation fails
        RuntimeError: If attack generation fails
        
    Example:
        >>> sweep_df = run_epsilon_sweep(
        ...     rf, iso_forest, surrogate, X_test, y_test,
        ...     clip_values=(0, 1),
        ...     eps_values=[0.01, 0.05, 0.1, 0.2],
        ...     ensemble_defense_predict_func=ensemble_defense_predict,
        ...     sample_size=100
        ... )
        >>> print(sweep_df)
           epsilon  Baseline  ZT-Shield (Defended)
        0     0.01      0.95                  0.97
        1     0.05      0.85                  0.92
        ...
    """
    # Input validation
    validate_sweep_inputs(X_test, y_test, clip_values, sample_size, eps_values)
    
    # Ensure float32 for PyTorch compatibility
    X_test = X_test.astype(np.float32)

    logger.info(
        f"Initiating Epsilon Sweep across {len(eps_values)} values "
        f"(Sample Size: {sample_size}, Random State: {random_state})"
    )
    
    try:
        # Create PyTorch classifier (once, outside loop for efficiency)
        classifier = PyTorchClassifier(
            model=surrogate_model,
            loss=nn.CrossEntropyLoss(),
            optimizer=optim.Adam(surrogate_model.parameters(), lr=0.001),
            input_shape=(X_test.shape[1],),
            nb_classes=2,
            clip_values=clip_values
        )
        
        # Use fixed sample for consistent curve comparison
        rng = np.random.RandomState(random_state)
        indices = rng.permutation(len(X_test))[:sample_size]
        X_sample = X_test[indices].copy()  # Explicit copy
        y_sample = y_test[indices].copy()
        
        logger.debug(f"Selected {len(X_sample)} samples for sweep")
        
        # Pre-allocate results storage for efficiency
        sweep_results = []
        
        # Perform sweep
        for idx, eps in enumerate(eps_values):
            try:
                # Generate adversarial examples
                attack = FastGradientMethod(estimator=classifier, eps=eps)
                X_adv = attack.generate(x=X_sample)
                
                # Validate adversarial examples
                if X_adv is None or len(X_adv) != sample_size:
                    raise RuntimeError(
                        f"Attack failed to generate {sample_size} examples at eps={eps}"
                    )
                
                # Baseline accuracy
                preds_baseline = rf_model.predict(X_adv)
                acc_baseline = accuracy_score(y_sample, preds_baseline)
                
                # Defended accuracy using ensemble
                preds_defended = safe_ensemble_predict(
                    rf_model, 
                    iso_forest, 
                    X_adv,
                    ensemble_defense_predict_func
                )
                acc_defended = accuracy_score(y_sample, preds_defended)
                
                # Store results
                sweep_results.append({
                    "epsilon": eps,
                    "Baseline": acc_baseline,
                    "ZT-Shield (Defended)": acc_defended
                })
                
                # Conditional logging (avoid overhead in production)
                if enable_debug_logging or logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"Eps {eps:.4f} ({idx+1}/{len(eps_values)}) | "
                        f"Baseline: {acc_baseline*100:.1f}% | "
                        f"Defended: {acc_defended*100:.1f}%"
                    )
                else:
                    logger.info(
                        f"Progress: {idx+1}/{len(eps_values)} epsilon values completed"
                    )
                
            except Exception as e:
                logger.error(f"Failed at epsilon={eps}: {str(e)}")
                # Option: Continue with NaN or re-raise
                # For now, we'll add NaN values and continue
                sweep_results.append({
                    "epsilon": eps,
                    "Baseline": np.nan,
                    "ZT-Shield (Defended)": np.nan
                })
        
        # Convert to DataFrame (efficient single operation)
        results_df = pd.DataFrame(sweep_results)
        
        logger.info(f"Epsilon Sweep Complete. Generated {len(results_df)} data points.")
        
        # Log summary statistics
        if len(results_df) > 0:
            baseline_mean = results_df['Baseline'].mean()
            defended_mean = results_df['ZT-Shield (Defended)'].mean()
            logger.info(
                f"Mean Accuracy - Baseline: {baseline_mean*100:.1f}%, "
                f"Defended: {defended_mean*100:.1f}%"
            )
        
        return results_df
        
    except Exception as e:
        logger.error(f"Epsilon sweep failed: {str(e)}")
        raise RuntimeError(f"Epsilon sweep failed: {str(e)}") from e


def plot_robustness_curve(
    sweep_df: pd.DataFrame,
    title: str = "Adversarial Robustness Curve",
    save_path: Optional[str] = None
) -> None:
    """
    Plot robustness curve from sweep results.
    
    Args:
        sweep_df: DataFrame from run_epsilon_sweep
        title: Plot title
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(
            sweep_df['epsilon'], 
            sweep_df['Baseline'], 
            marker='o', 
            label='Baseline',
            linewidth=2
        )
        plt.plot(
            sweep_df['epsilon'], 
            sweep_df['ZT-Shield (Defended)'], 
            marker='s', 
            label='ZT-Shield (Defended)',
            linewidth=2
        )
        
        plt.xlabel('Epsilon (Perturbation Magnitude)', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
        
    except ImportError:
        logger.warning("matplotlib not available, skipping plot generation")
