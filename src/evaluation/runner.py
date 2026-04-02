"""
Research evaluation runner module.

Orchestrates robust multi-seed evaluation suites for adversarial attack assessment.
Handles model loading, attack execution, defense evaluation, and statistical analysis.
"""

import os
import joblib
import pandas as pd
import numpy as np
import logging
from typing import Tuple, List, Dict, Any, Callable, Optional
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)

# Import from fixed modules
try:
    from src.config import MODEL_DIR
    from src.core.defense import ensemble_defense_predict
    from src.core.metrics import calculate_evasion_rate, get_perturbation_norms
    from src.core.utils import set_seed
    from src.evaluation.statistics import calculate_statistical_significance, calculate_confidence_interval
except ImportError as e:
    logger.warning(f"Import failed, using fallbacks: {e}")
    MODEL_DIR = "models"


def validate_model_assets(model_path: str, iso_path: str, test_path: str) -> None:
    """
    Validate that all required model files exist.
    
    Args:
        model_path: Path to main model
        iso_path: Path to isolation forest
        test_path: Path to test set
        
    Raises:
        FileNotFoundError: If any file is missing
    """
    files_to_check = {
        'Main Model': model_path,
        'Isolation Forest': iso_path,
        'Test Set': test_path
    }
    
    missing = []
    for name, path in files_to_check.items():
        if not os.path.exists(path):
            missing.append(f"{name} ({path})")
    
    if missing:
        raise FileNotFoundError(
            f"Missing required files:\n" + "\n".join(f"  - {m}" for m in missing)
        )


def load_system_assets(
    model_name: str = "random_forest.pkl"
) -> Tuple[Any, Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Load models and data from the centralized model directory.
    
    Args:
        model_name: Name of the main classifier model file
        
    Returns:
        Tuple of:
            - rf: Random Forest model
            - iso: Isolation Forest model
            - X_test: Test features
            - y_test: Test labels
            - X_train: Training features
            - y_train: Training labels
            - clip_values: Tuple of (min_bounds, max_bounds) for each feature
            
    Raises:
        FileNotFoundError: If required files don't exist
        ValueError: If data is invalid
    """
    logger.info(f"Loading system assets from {MODEL_DIR}...")
    
    # Construct paths
    rf_path = os.path.join(MODEL_DIR, model_name)
    iso_path = os.path.join(MODEL_DIR, "isolation_forest.pkl")
    test_path = os.path.join(MODEL_DIR, "test_set.csv")
    train_path = os.path.join(MODEL_DIR, "train_set.csv")
    bounds_path = os.path.join(MODEL_DIR, "feature_bounds.pkl")
    
    # Validate existence
    validate_model_assets(rf_path, iso_path, test_path)
    
    try:
        # Load models
        rf = joblib.load(rf_path)
        iso = joblib.load(iso_path)
        logger.debug(f"Loaded models: {model_name}, isolation_forest.pkl")
        
        # Load test set
        test_df = pd.read_csv(test_path)
        if 'label' not in test_df.columns:
            raise ValueError("Test set missing 'label' column")
        
        X_test = test_df.drop(columns=['label']).values.astype(np.float32)
        y_test = test_df['label'].values
        
        logger.debug(f"Loaded test set: {X_test.shape}")
        
        # Load training set (for surrogate)
        if os.path.exists(train_path):
            train_df = pd.read_csv(train_path)
            X_train = train_df.drop(columns=['label']).values.astype(np.float32)
            y_train = train_df['label'].values
            logger.debug(f"Loaded training set: {X_train.shape}")
        else:
            logger.warning("Training set not found, using empty arrays")
            X_train = np.array([]).astype(np.float32)
            y_train = np.array([])
        
        # Load feature bounds
        if os.path.exists(bounds_path):
            clip_values = joblib.load(bounds_path)
            logger.debug(f"Loaded feature bounds: {type(clip_values)}")
        else:
            logger.warning("Feature bounds not found, using min/max from test data")
            clip_values = (X_test.min(axis=0), X_test.max(axis=0))
        
        # Validate data
        if len(X_test) == 0:
            raise ValueError("Test set is empty")
        
        if len(X_test) != len(y_test):
            raise ValueError(f"X_test and y_test length mismatch: {len(X_test)} != {len(y_test)}")
        
        if not np.isfinite(X_test).all():
            raise ValueError("Test set contains non-finite values")
        
        logger.info(f"Successfully loaded all assets")
        return rf, iso, X_test, y_test, X_train, y_train, clip_values
        
    except Exception as e:
        logger.error(f"Failed to load system assets: {str(e)}")
        raise RuntimeError(f"Asset loading failed: {str(e)}") from e


def evaluate_attack_vector(
    rf,
    iso,
    X_orig: np.ndarray,
    X_adv: np.ndarray,
    y_sample: np.ndarray
) -> Dict[str, float]:
    """
    Compute professional research metrics for a single attack run.
    
    Args:
        rf: Random Forest model
        iso: Isolation Forest model
        X_orig: Original samples
        X_adv: Adversarial samples
        y_sample: True labels
        
    Returns:
        Dictionary with metrics:
            - acc_base: Baseline accuracy
            - acc_def: Defended accuracy
            - evasion_base: Baseline evasion rate
            - evasion_def: Defended evasion rate
            - l2: Average L2 perturbation
            - linf: Average L-infinity perturbation
            - latency_ms: Defense latency in milliseconds
    """
    try:
        # Validate inputs
        if len(X_orig) != len(X_adv) or len(X_orig) != len(y_sample):
            raise ValueError(
                f"Length mismatch: X_orig={len(X_orig)}, "
                f"X_adv={len(X_adv)}, y_sample={len(y_sample)}"
            )
        
        # Baseline results (no defense)
        preds_base = rf.predict(X_adv)
        acc_base = accuracy_score(y_sample, preds_base)
        evasion_base = calculate_evasion_rate(y_sample, preds_base)
        
        # Defended results (with ensemble defense)
        # Use return_latency=True to get both predictions and timing
        preds_def, latency_ms = ensemble_defense_predict(
            rf, iso, X_adv, return_latency=True
        )
        
        acc_def = accuracy_score(y_sample, preds_def)
        evasion_def = calculate_evasion_rate(y_sample, preds_def)
        
        # Perturbation norms
        l2, linf = get_perturbation_norms(X_orig, X_adv)
        
        results = {
            "acc_base": float(acc_base),
            "acc_def": float(acc_def),
            "evasion_base": float(evasion_base),
            "evasion_def": float(evasion_def),
            "l2": float(l2),
            "linf": float(linf),
            "latency_ms": float(latency_ms)
        }
        
        logger.debug(
            f"Attack evaluation: "
            f"Base_Acc={acc_base:.3f}, "
            f"Def_Acc={acc_def:.3f}, "
            f"Evasion_Base={evasion_base:.3f}, "
            f"Evasion_Def={evasion_def:.3f}"
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Attack evaluation failed: {str(e)}")
        raise RuntimeError(f"Evaluation failed: {str(e)}") from e


def run_research_suite(
    attack_func: Callable,
    rf,
    iso,
    X_test: np.ndarray,
    y_test: np.ndarray,
    clip_values: Tuple,
    multi_seed: bool = False,
    seeds: List[int] = [42, 43, 44],
    **kwargs
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    Orchestrate a robust multi-seed evaluation suite.
    
    This function runs the same attack multiple times with different random
    seeds to ensure statistical rigor and reproducibility.
    
    Args:
        attack_func: Attack function to run. Must return:
            - Minimum: (X_adv, X_sample, y_sample)
            - Optional: Additional values like query counts
        rf: Random Forest model
        iso: Isolation Forest model
        X_test: Test features
        y_test: Test labels
        clip_values: Feature bounds
        multi_seed: If True, run with multiple seeds
        seeds: List of random seeds to use
        **kwargs: Additional arguments passed to attack_func
        
    Returns:
        Tuple of:
            - summary: Dictionary of aggregate statistics
            - results_log: List of per-run results
            
    Example:
        >>> from src.attacks.blackbox import run_blackbox_attack
        >>> summary, log = run_research_suite(
        ...     run_blackbox_attack, rf, iso, X_test, y_test, clip_values,
        ...     multi_seed=True, sample_size=100
        ... )
        >>> print(f"Mean evasion rate: {summary['mean_evasion_def']:.3f}")
    """
    logger.info(
        f"Starting research suite: "
        f"multi_seed={multi_seed}, "
        f"num_seeds={len(seeds) if multi_seed else 1}"
    )
    
    results_log = []
    failed_runs = []
    active_seeds = seeds if multi_seed else [seeds[0]]
    
    for seed_idx, seed in enumerate(active_seeds):
        logger.info(f"Running evaluation {seed_idx + 1}/{len(active_seeds)} (seed={seed})")
        
        try:
            # Set random seed for reproducibility
            set_seed(seed)
            
            # Run attack function
            # Handle variable return values:
            # - Minimum: (X_adv, X_sample, y_sample)
            # - Extended: (X_adv, X_sample, y_sample, extra_metric1, extra_metric2, ...)
            attack_results = attack_func(
                rf, X_test, y_test, clip_values, 
                random_state=seed,  # Pass seed to attack
                **kwargs
            )
            
            # Unpack minimum required values
            if not isinstance(attack_results, tuple) or len(attack_results) < 3:
                raise ValueError(
                    f"Attack function must return at least (X_adv, X_sample, y_sample), "
                    f"got {type(attack_results)} with length {len(attack_results) if isinstance(attack_results, tuple) else 'N/A'}"
                )
            
            X_adv = attack_results[0]
            X_sample = attack_results[1]
            y_sample = attack_results[2]
            extra_metrics = attack_results[3:] if len(attack_results) > 3 else []
            
            # Validate attack outputs
            if X_adv is None or X_sample is None or y_sample is None:
                raise ValueError("Attack returned None values")
            
            # Evaluate the attack
            res = evaluate_attack_vector(rf, iso, X_sample, X_adv, y_sample)
            
            # Add extra metrics if present
            # Common extra metrics: avg_queries, total_queries, etc.
            if len(extra_metrics) >= 1:
                res["avg_queries"] = float(extra_metrics[0])
            if len(extra_metrics) >= 2:
                res["total_queries"] = float(extra_metrics[1])
            
            # Add seed to results
            res["seed"] = seed
            
            results_log.append(res)
            logger.info(
                f"Seed {seed} complete: "
                f"evasion_def={res['evasion_def']:.3f}, "
                f"acc_def={res['acc_def']:.3f}"
            )
            
        except Exception as e:
            logger.error(f"Seed {seed} failed: {str(e)}")
            failed_runs.append({"seed": seed, "error": str(e)})
            
            # If not multi-seed, re-raise immediately
            if not multi_seed:
                raise RuntimeError(f"Single-seed evaluation failed: {str(e)}") from e
    
    # Check if we have any successful runs
    if len(results_log) == 0:
        raise RuntimeError(
            f"All {len(active_seeds)} evaluation runs failed. "
            f"Errors: {failed_runs}"
        )
    
    if len(failed_runs) > 0:
        logger.warning(
            f"{len(failed_runs)}/{len(active_seeds)} runs failed: "
            f"{[f['seed'] for f in failed_runs]}"
        )
    
    # Aggregate statistics across runs
    summary = {
        "num_runs": len(results_log),
        "num_failed": len(failed_runs),
        "mean_evasion_base": float(np.mean([r['evasion_base'] for r in results_log])),
        "mean_evasion_def": float(np.mean([r['evasion_def'] for r in results_log])),
        "std_evasion_def": float(np.std([r['evasion_def'] for r in results_log])),
        "mean_robust_acc_def": float(np.mean([r['acc_def'] for r in results_log])),
        "std_acc_def": float(np.std([r['acc_def'] for r in results_log])),
        "mean_latency_ms": float(np.mean([r['latency_ms'] for r in results_log])),
        "mean_l2": float(np.mean([r['l2'] for r in results_log])),
        "mean_linf": float(np.mean([r['linf'] for r in results_log]))
    }
    
    # Add query statistics if available
    if 'avg_queries' in results_log[0]:
        summary["mean_avg_queries"] = float(np.mean([r['avg_queries'] for r in results_log]))
    
    # Add statistical rigor (T-test) if multi-seed
    if multi_seed and len(results_log) >= 2:
        try:
            base_evasions = [r['evasion_base'] for r in results_log]
            def_evasions = [r['evasion_def'] for r in results_log]
            
            stats = calculate_statistical_significance(base_evasions, def_evasions)
            summary.update(stats)
            
            ci_margin = calculate_confidence_interval(def_evasions)
            summary["ci_95_margin"] = float(ci_margin)
            summary["ci_95_lower"] = float(summary["mean_evasion_def"] - ci_margin)
            summary["ci_95_upper"] = float(summary["mean_evasion_def"] + ci_margin)
            
            logger.info(
                f"Statistical analysis: "
                f"p_value={stats['p_value']:.4f}, "
                f"significant={stats['is_significant']}"
            )
        except Exception as e:
            logger.warning(f"Statistical analysis failed: {str(e)}")
    
    logger.info(
        f"Research suite complete: "
        f"{len(results_log)} successful runs, "
        f"mean_evasion_def={summary['mean_evasion_def']:.3f}"
    )
    
    return summary, results_log
