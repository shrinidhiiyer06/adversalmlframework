"""
White-box transfer attack implementation using Fast Gradient Method (FGM).

Uses a surrogate model to generate transferable adversarial examples that can fool the target model.
"""

import numpy as np
from typing import Tuple, Optional
import torch.nn as nn
import torch.optim as optim
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
import logging

# Note: Adjust this import to match your project structure
# from src.config import FGM_EPS

logger = logging.getLogger(__name__)

# Default epsilon value if not provided via config
DEFAULT_FGM_EPS = 0.1


def validate_whitebox_inputs(
    X_test: np.ndarray,
    y_test: np.ndarray,
    sample_size: int,
    clip_values: Tuple[float, float],
    eps: float
) -> None:
    """
    Validate input parameters for white-box attack.
    
    Args:
        X_test: Test features
        y_test: Test labels
        sample_size: Number of samples to attack
        clip_values: Min/max values for clipping
        eps: Perturbation magnitude
        
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
    
    if np.any(np.array(eps) < 0):
        raise ValueError(f"eps must be non-negative, got {eps}")
    
    if np.any(np.array(eps) > (clip_values[1] - clip_values[0])):
        logger.warning(
            f"eps ({eps}) is larger than clip range for some features. "
            "This may lead to heavily clipped perturbations."
        )
    
    if not np.isfinite(X_test).all():
        raise ValueError("X_test contains non-finite values (NaN or inf)")


def run_whitebox_attack(
    surrogate_model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    clip_values: Tuple[float, float],
    sample_size: int = 100,
    eps: Optional[float] = None,
    random_state: Optional[int] = None,
    norm: str = 'inf',
    minimal: bool = False,
    batch_size: int = 128
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run white-box transfer attack using Fast Gradient Method (FGM).
    
    Generates adversarial examples by computing gradients on a surrogate model
    and applying them to create perturbations. These examples often transfer
    to other models due to shared decision boundaries.
    
    FGM (also known as FGSM - Fast Gradient Sign Method when norm='inf') is
    a computationally efficient attack that perturbs inputs in the direction
    of the gradient of the loss function.
    
    Args:
        surrogate_model: PyTorch model to compute gradients (must be in eval mode)
        X_test: Test features of shape (n_samples, n_features)
        y_test: Test labels of shape (n_samples,)
        clip_values: Tuple of (min, max) values for clipping adversarial examples
        sample_size: Number of samples to attack (default: 100)
        eps: Perturbation magnitude (L-infinity norm bound). If None, uses DEFAULT_FGM_EPS
        random_state: Random seed for reproducibility (default: None)
        norm: Norm to use for perturbation ('inf', 1, 2). Default: 'inf' (FGSM)
        minimal: If True, computes minimal perturbation (slower but more precise)
        batch_size: Batch size for adversarial generation (default: 128)
        
    Returns:
        Tuple containing:
            - X_adv: Adversarial examples of shape (sample_size, n_features)
            - X_sample: Original samples of shape (sample_size, n_features)
            - y_sample: Original labels of shape (sample_size,)
            
    Raises:
        ValueError: If input validation fails
        RuntimeError: If attack generation fails
        
    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> 
        >>> # Define surrogate model
        >>> surrogate = nn.Sequential(
        ...     nn.Linear(10, 50),
        ...     nn.ReLU(),
        ...     nn.Linear(50, 2)
        ... )
        >>> 
        >>> # Run attack
        >>> X_adv, X_orig, y_orig = run_whitebox_attack(
        ...     surrogate, X_test, y_test,
        ...     clip_values=(0, 1),
        ...     sample_size=50,
        ...     eps=0.1
        ... )
        >>> print(f"Generated {len(X_adv)} adversarial examples")
        
    Notes:
        - The surrogate model does NOT need to be trained on the target task,
          but better surrogate training often leads to better transfer
        - FGM is fast but may not be as effective as iterative methods
        - Higher epsilon values create stronger perturbations but are more detectable
        - The optimizer parameter in PyTorchClassifier is not used for FGM
          (no iterative optimization needed)
    """
    # Use default epsilon if not provided
    if eps is None:
        eps = DEFAULT_FGM_EPS
        logger.info(f"Using default epsilon: {eps}")
    
    # Input validation
    validate_whitebox_inputs(X_test, y_test, sample_size, clip_values, eps)
    
    # Ensure float32 for PyTorch compatibility
    X_test = X_test.astype(np.float32)
    
    logger.info(
        f"Initiating White-Box FGM Attack (Transfer Attack) "
        f"[Epsilon: {eps}, Sample Size: {sample_size}, Norm: {norm}]"
    )
    
    try:
        # Ensure surrogate model is in evaluation mode
        if hasattr(surrogate_model, 'eval'):
            surrogate_model.eval()
        
        # Create PyTorch classifier wrapper for ART
        # Note: optimizer is created but not used by FGM (non-iterative attack)
        # We keep it for ART compatibility, but use a lightweight dummy optimizer
        classifier = PyTorchClassifier(
            model=surrogate_model,
            loss=nn.CrossEntropyLoss(),
            optimizer=None,  # Not needed for FGM, saves memory
            input_shape=(X_test.shape[1],),
            nb_classes=2,
            clip_values=clip_values
        )
        
        # Sample selection with controlled randomness
        if sample_size >= len(X_test):
            indices = np.arange(len(X_test))
            sample_size = len(X_test)
        else:
            rng = np.random.RandomState(random_state)
            indices = rng.permutation(len(X_test))[:sample_size]
        
        X_sample = X_test[indices].copy()  # Explicit copy to avoid view issues
        y_sample = y_test[indices].copy()
        
        logger.debug(f"Selected {len(X_sample)} samples for attack")
        
        # Initialize FGM attack
        attack = FastGradientMethod(
            estimator=classifier,
            eps=eps,
            eps_step=eps,
            norm=norm,
            minimal=minimal,
            batch_size=batch_size
        )
        
        # Generate adversarial examples
        logger.debug("Generating adversarial examples...")
        X_adv = attack.generate(x=X_sample)
        
        # Validate output
        if X_adv is None or len(X_adv) != sample_size:
            raise RuntimeError(
                f"Attack failed to generate {sample_size} adversarial examples. "
                f"Got {len(X_adv) if X_adv is not None else 0}"
            )
        
        # Compute perturbation statistics
        perturbations = np.abs(X_adv - X_sample)
        if norm == 'inf':
            max_pert = np.max(perturbations)
            avg_pert = np.mean(np.max(perturbations, axis=1))
            logger.info(
                f"Attack Complete. L-inf Perturbation - Max: {max_pert:.6f}, "
                f"Avg Max: {avg_pert:.6f}"
            )
        elif norm == 2:
            avg_l2 = np.mean(np.linalg.norm(perturbations, axis=1))
            logger.info(f"Attack Complete. Avg L2 Perturbation: {avg_l2:.6f}")
        else:
            avg_l1 = np.mean(np.linalg.norm(perturbations, ord=1, axis=1))
            logger.info(f"Attack Complete. Avg L1 Perturbation: {avg_l1:.6f}")
        
        # Verify adversarial examples are within bounds
        if not (X_adv >= clip_values[0]).all() or not (X_adv <= clip_values[1]).all():
            logger.warning(
                "Some adversarial examples exceeded clip bounds. "
                "This should not happen - possible numerical issues."
            )
        
        return X_adv, X_sample, y_sample
        
    except Exception as e:
        logger.error(
            f"White-box attack failed: {str(e)} "
            f"(sample_size={sample_size}, eps={eps}, norm={norm})"
        )
        raise RuntimeError(f"White-box attack failed: {str(e)}") from e


def evaluate_transferability(
    X_adv: np.ndarray,
    X_orig: np.ndarray,
    y_true: np.ndarray,
    surrogate_model,
    target_model,
    clip_values: Tuple[float, float]
) -> dict:
    """
    Evaluate how well adversarial examples transfer from surrogate to target.
    
    Args:
        X_adv: Adversarial examples
        X_orig: Original examples
        y_true: True labels
        surrogate_model: Model used to generate adversarial examples
        target_model: Target model to fool
        clip_values: Clipping bounds
        
    Returns:
        Dictionary with transferability metrics:
            - surrogate_success_rate: Attack success on surrogate
            - target_success_rate: Attack success on target (transferability)
            - surrogate_accuracy: Accuracy on adversarial examples
            - target_accuracy: Accuracy on adversarial examples
    """
    from sklearn.metrics import accuracy_score
    
    # Get predictions from surrogate
    if hasattr(surrogate_model, 'predict'):
        surrogate_preds_orig = surrogate_model.predict(X_orig)
        surrogate_preds_adv = surrogate_model.predict(X_adv)
    else:
        # PyTorch model - need to convert and handle
        import torch
        surrogate_model.eval()
        with torch.no_grad():
            X_orig_tensor = torch.FloatTensor(X_orig)
            X_adv_tensor = torch.FloatTensor(X_adv)
            surrogate_preds_orig = torch.argmax(surrogate_model(X_orig_tensor), dim=1).numpy()
            surrogate_preds_adv = torch.argmax(surrogate_model(X_adv_tensor), dim=1).numpy()
    
    # Get predictions from target
    target_preds_orig = target_model.predict(X_orig)
    target_preds_adv = target_model.predict(X_adv)
    
    # Calculate metrics
    surrogate_success = np.mean(surrogate_preds_orig != surrogate_preds_adv)
    target_success = np.mean(target_preds_orig != target_preds_adv)
    
    surrogate_acc = accuracy_score(y_true, surrogate_preds_adv)
    target_acc = accuracy_score(y_true, target_preds_adv)
    
    return {
        'surrogate_success_rate': surrogate_success,
        'target_success_rate': target_success,
        'surrogate_accuracy': surrogate_acc,
        'target_accuracy': target_acc,
        'transferability_ratio': target_success / surrogate_success if surrogate_success > 0 else 0
    }
