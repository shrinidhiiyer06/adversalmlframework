"""
Black-box adversarial attack module using HopSkipJump.

This module implements query-efficient black-box attacks with comprehensive
query tracking for adversarial robustness evaluation.
"""

import numpy as np
from typing import Tuple, Optional
from art.attacks.evasion import HopSkipJump
from art.estimators.classification.scikitlearn import ScikitlearnClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import logging

logger = logging.getLogger(__name__)


class QueryCountingWrapper(BaseEstimator, ClassifierMixin):
    """
    Wraps a model to count total queries during an attack.
    
    Implements the full sklearn estimator interface for compatibility with
    ART and other frameworks that expect standard sklearn behavior.
    
    Attributes:
        model: The wrapped classifier model
        query_count: Total number of queries made to the model
        classes_: Class labels (proxied from wrapped model)
        n_features_in_: Number of input features (proxied from wrapped model)
    """
    
    def __init__(self, model):
        """
        Initialize the query counting wrapper.
        
        Args:
            model: A trained sklearn-compatible classifier
        """
        self.model = model
        self.query_count = 0
        
        # Safely proxy essential attributes for ART/sklearn compatibility
        self.classes_ = getattr(model, 'classes_', np.array([0, 1]))
        if hasattr(model, 'classes_'):
            self.nb_classes = len(model.classes_)
        else:
            self.nb_classes = 2 # Default binary
            
        self.n_features_in_ = getattr(model, 'n_features_in_', None)
        self.feature_importances_ = getattr(model, 'feature_importances_', None)
        
        # Store original model parameters
        self._estimator_type = getattr(model, '_estimator_type', 'classifier')

    def fit(self, X, y, **kwargs):
        """
        Fit the underlying model (not typically used in attack scenarios).
        
        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional arguments passed to underlying model
            
        Returns:
            self
        """
        self.model.fit(X, y, **kwargs)
        
        # Update proxied attributes after fitting
        self.classes_ = getattr(self.model, 'classes_', self.classes_)
        self.n_features_in_ = getattr(self.model, 'n_features_in_', X.shape[1])
        
        return self

    def predict(self, X):
        """
        Make predictions and increment query counter.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Predicted class labels of shape (n_samples,)
        """
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        
        self.query_count += len(X)
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities and increment query counter.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Class probabilities of shape (n_samples, n_classes)
        """
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        
        self.query_count += len(X)
        return self.model.predict_proba(X)
    
    def score(self, X, y, sample_weight=None):
        """
        Calculate accuracy score (does not increment query counter for fairness).
        
        Args:
            X: Test features
            y: True labels
            sample_weight: Optional sample weights
            
        Returns:
            Accuracy score
        """
        return self.model.score(X, y, sample_weight=sample_weight)
    
    def get_params(self, deep=True):
        """Get parameters (sklearn compatibility)."""
        return {'model': self.model}
    
    def set_params(self, **params):
        """Set parameters (sklearn compatibility)."""
        if 'model' in params:
            self.model = params['model']
        return self
    
    def reset_query_count(self):
        """Reset the query counter to zero."""
        self.query_count = 0


def validate_inputs(X_test: np.ndarray, 
                   y_test: np.ndarray, 
                   sample_size: int, 
                   clip_values: Tuple[float, float],
                   max_iter: int) -> None:
    """
    Validate input parameters for the black-box attack.
    
    Args:
        X_test: Test features
        y_test: Test labels
        sample_size: Number of samples to attack
        clip_values: Min/max values for clipping adversarial examples
        max_iter: Maximum iterations for the attack
        
    Raises:
        ValueError: If any validation check fails
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
    
    if max_iter < 1:
        raise ValueError(f"max_iter must be positive, got {max_iter}")
    
    if not np.isfinite(X_test).all():
        raise ValueError("X_test contains non-finite values (NaN or inf)")


def run_blackbox_attack(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    clip_values: Tuple[float, float],
    sample_size: int = 100,
    max_iter: int = 50,
    max_eval: int = 100,
    init_eval: int = 10,
    random_state: Optional[int] = None,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    """
    Run HopSkipJump black-box attack with query complexity tracking.
    
    HopSkipJump is a decision-based attack that only requires access to
    model predictions (not gradients or probabilities), making it effective
    against black-box models.
    
    Args:
        model: Trained classifier with predict/predict_proba methods
        X_test: Test features of shape (n_samples, n_features)
        y_test: Test labels of shape (n_samples,)
        clip_values: Tuple of (min, max) values for clipping adversarial examples
        sample_size: Number of samples to attack (default: 100)
        max_iter: Maximum iterations per sample (default: 50)
        max_eval: Maximum evaluations per iteration (default: 100)
        init_eval: Initial evaluations for boundary search (default: 10)
        random_state: Random seed for reproducibility (default: None)
        verbose: Whether to print attack progress (default: False)
        
    Returns:
        Tuple containing:
            - X_adv: Adversarial examples of shape (sample_size, n_features)
            - X_sample: Original samples of shape (sample_size, n_features)
            - y_sample: Original labels of shape (sample_size,)
            - avg_queries: Average queries per sample
            - total_queries: Total queries made during attack
            
    Raises:
        ValueError: If input validation fails
        RuntimeError: If attack fails to generate adversarial examples
        
    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier().fit(X_train, y_train)
        >>> X_adv, X_orig, y_orig, avg_q, total_q = run_blackbox_attack(
        ...     model, X_test, y_test, clip_values=(0, 1), sample_size=50
        ... )
        >>> print(f"Generated {len(X_adv)} adversarial examples")
        >>> print(f"Average queries: {avg_q:.1f}")
    """
    # Input validation
    validate_inputs(X_test, y_test, sample_size, clip_values, max_iter)
    
    logger.info(
        f"Initiating Black-Box HSJ Attack "
        f"(Sample Size: {sample_size}, Max Iter: {max_iter}, "
        f"Max Eval: {max_eval}, Init Eval: {init_eval})"
    )
    
    try:
        # Wrap model for query counting
        policed_model = QueryCountingWrapper(model)
        classifier = ScikitlearnClassifier(
            model=policed_model, 
            clip_values=clip_values
        )
        
        # Sample selection with controlled randomness
        rng = np.random.RandomState(random_state)
        indices = rng.permutation(len(X_test))[:sample_size]
        X_sample = X_test[indices].copy()  # Explicit copy to avoid view issues
        y_sample = y_test[indices].copy()
        
        # Initialize attack
        attack = HopSkipJump(
            classifier=classifier,
            max_iter=max_iter,
            max_eval=max_eval,
            init_eval=init_eval,
            verbose=verbose
        )
        
        # Generate adversarial examples
        logger.debug(f"Generating adversarial examples for {sample_size} samples...")
        X_adv = attack.generate(x=X_sample)
        
        # Query statistics
        total_queries = policed_model.query_count
        avg_queries = total_queries / sample_size if sample_size > 0 else 0
        
        logger.info(
            f"Attack Complete. Total Queries: {total_queries} "
            f"(Avg: {avg_queries:.1f} per sample)"
        )
        
        # Validate output
        if X_adv is None or len(X_adv) != sample_size:
            raise RuntimeError(
                f"Attack failed to generate {sample_size} adversarial examples. "
                f"Got {len(X_adv) if X_adv is not None else 0}"
            )
        
        return X_adv, X_sample, y_sample, avg_queries, total_queries
        
    except Exception as e:
        logger.error(
            f"Black-box attack failed: {str(e)} "
            f"(sample_size={sample_size}, max_iter={max_iter})"
        )
        raise RuntimeError(f"Black-box attack failed: {str(e)}") from e
