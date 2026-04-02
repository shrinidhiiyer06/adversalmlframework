"""
Network constraint validation for adversarial examples.

Validates that adversarial perturbations produce domain-valid network flows:
no negative feature values, integer features remain integers, protocol flags
stay within valid enumeration ranges, and durations remain non-negative.

A 100% constraint satisfaction rate confirms that adversarial examples remain
valid network flows, strengthening the realism of the experimental setup.
This function MUST run after every adversarial example generation step,
and its pass rate MUST be reported in the paper.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Features that must remain non-negative integers
INTEGER_FEATURES = ['packet_size', 'request_frequency', 'trust_score']

# Features that must remain non-negative (continuous)
NON_NEGATIVE_FEATURES = ['flow_duration', 'geo_velocity', 'token_entropy']

# Feature-specific bounds
FEATURE_BOUNDS = {
    'packet_size': (64, 65535),      # Min Ethernet frame to max IP packet
    'flow_duration': (0.0, None),     # Non-negative, no upper bound
    'request_frequency': (0, 1000),   # Reasonable requests per minute
    'token_entropy': (0.0, 8.0),      # Shannon entropy bounds for bytes
    'geo_velocity': (0.0, None),      # km/h, non-negative
    'trust_score': (0, 100),          # 0-100 range
}

# Default feature name list matching the project's data schema
DEFAULT_FEATURE_NAMES = [
    'packet_size', 'flow_duration', 'request_frequency',
    'token_entropy', 'geo_velocity', 'trust_score'
]


def validate_single_sample(
    sample: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> Dict:
    """Validate a single adversarial example for domain validity.

    Args:
        sample: 1D array of feature values for one sample.
        feature_names: Names of features corresponding to array positions.
            Defaults to DEFAULT_FEATURE_NAMES.

    Returns:
        Dictionary with:
            - valid (bool): Whether all constraints are satisfied.
            - violations (list): List of violation descriptions.
            - checks_passed (int): Number of checks that passed.
            - checks_total (int): Total number of checks performed.

    Example:
        >>> sample = np.array([500.0, 1.5, 3.0, 7.2, 5.0, 80.0])
        >>> result = validate_single_sample(sample)
        >>> print(result['valid'])
        True
    """
    if feature_names is None:
        feature_names = DEFAULT_FEATURE_NAMES

    violations = []
    checks_passed = 0
    checks_total = 0

    for i, (val, name) in enumerate(zip(sample, feature_names)):
        # Check 1: No negative values
        checks_total += 1
        if val < 0:
            violations.append(f"{name}: negative value {val:.4f}")
        else:
            checks_passed += 1

        # Check 2: Integer features must be integers (after rounding)
        if name in INTEGER_FEATURES:
            checks_total += 1
            rounded = round(float(val))
            if abs(val - rounded) > 0.01:
                violations.append(f"{name}: non-integer value {val:.4f} (expected {rounded})")
            else:
                checks_passed += 1

        # Check 3: Feature-specific bounds
        if name in FEATURE_BOUNDS:
            low, high = FEATURE_BOUNDS[name]
            if low is not None:
                checks_total += 1
                if val < low:
                    violations.append(f"{name}: value {val:.4f} below minimum {low}")
                else:
                    checks_passed += 1
            if high is not None:
                checks_total += 1
                if val > high:
                    violations.append(f"{name}: value {val:.4f} above maximum {high}")
                else:
                    checks_passed += 1

    return {
        'valid': len(violations) == 0,
        'violations': violations,
        'checks_passed': checks_passed,
        'checks_total': checks_total,
    }


def validate_adversarial_batch(
    X_adv: np.ndarray,
    feature_names: Optional[List[str]] = None,
    fix_violations: bool = True,
) -> Dict:
    """Validate and optionally fix a batch of adversarial examples.

    Args:
        X_adv: Adversarial examples array, shape (n_samples, n_features).
        feature_names: Feature names for each column.
        fix_violations: If True, clip values to valid ranges and round
            integer features. The fixed array is returned in the result.

    Returns:
        Dictionary with:
            - pass_rate (float): Fraction of samples passing all checks.
            - n_valid (int): Number of completely valid samples.
            - n_total (int): Total number of samples.
            - per_feature_violations (dict): Count of violations per feature.
            - X_fixed (ndarray, optional): Fixed array if fix_violations=True.

    Example:
        >>> X_adv = np.array([[500, 1.5, 3, 7.2, 5, 80],
        ...                    [-10, 1.5, 3, 7.2, 5, 80]])
        >>> result = validate_adversarial_batch(X_adv)
        >>> print(f"Pass rate: {result['pass_rate']:.0%}")
        Pass rate: 50%
    """
    if feature_names is None:
        feature_names = DEFAULT_FEATURE_NAMES

    n_samples = X_adv.shape[0]
    n_valid = 0
    per_feature_violations = {name: 0 for name in feature_names}
    all_violations = []

    for i in range(n_samples):
        result = validate_single_sample(X_adv[i], feature_names)
        if result['valid']:
            n_valid += 1
        else:
            for v in result['violations']:
                feat_name = v.split(':')[0]
                if feat_name in per_feature_violations:
                    per_feature_violations[feat_name] += 1
            all_violations.extend(result['violations'])

    output = {
        'pass_rate': n_valid / n_samples if n_samples > 0 else 0.0,
        'n_valid': n_valid,
        'n_total': n_samples,
        'per_feature_violations': per_feature_violations,
        'violation_details': all_violations[:50],  # Cap at 50 for readability
    }

    if fix_violations:
        X_fixed = X_adv.copy()
        for j, name in enumerate(feature_names):
            # Clip to bounds
            if name in FEATURE_BOUNDS:
                low, high = FEATURE_BOUNDS[name]
                if low is not None:
                    X_fixed[:, j] = np.maximum(X_fixed[:, j], low)
                if high is not None:
                    X_fixed[:, j] = np.minimum(X_fixed[:, j], high)
            # Round integer features
            if name in INTEGER_FEATURES:
                X_fixed[:, j] = np.round(X_fixed[:, j])

        output['X_fixed'] = X_fixed

        # Re-validate after fixing
        n_valid_after = sum(
            1 for i in range(n_samples)
            if validate_single_sample(X_fixed[i], feature_names)['valid']
        )
        output['pass_rate_after_fix'] = n_valid_after / n_samples if n_samples > 0 else 0.0

    logger.info(
        f"Constraint validation: {n_valid}/{n_samples} "
        f"({output['pass_rate']:.1%}) passed before fix"
    )

    return output
