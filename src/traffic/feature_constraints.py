import numpy as np

# Define realistic numeric bounds for NSL-KDD features
# These are used to clip adversarial perturbations to realistic ranges
FEATURE_BOUNDS = {
    "duration": (0, 1e5),
    "src_bytes": (0, 1e6),
    "dst_bytes": (0, 1e6),
    "wrong_fragment": (0, 3),
    "urgent": (0, 3),
    "hot": (0, 100),
    "num_failed_logins": (0, 5),
    "num_compromised": (0, 100),
    "num_root": (0, 100),
    "num_file_creations": (0, 100),
    "num_shells": (0, 10),
    "num_access_files": (0, 10),
    "count": (0, 511),
    "srv_count": (0, 511),
}

# Indices of categorical encoded features that should NOT be perturbed continuously
# protocol_type index: 1, service index: 2, flag index: 3
CATEGORICAL_INDICES = [1, 2, 3]

def apply_domain_constraints(x_adv_np, feature_names):
    """
    Apply domain-specific constraints to adversarial samples
    
    Args:
        x_adv_np: Adversarial samples (numpy array)
        feature_names: List of feature names corresponding to columns
        
    Returns:
        Constrained numpy array
    """
    constrained = x_adv_np.copy()
    
    for i, name in enumerate(feature_names):
        if name in FEATURE_BOUNDS:
            low, high = FEATURE_BOUNDS[name]
            constrained[:, i] = np.clip(constrained[:, i], low, high)
            
    # Ensure non-negative for all rate features (usually indices 24-40)
    # NSL-KDD rate features are usually [0, 1]
    rate_indices = list(range(24, 41))
    constrained[:, rate_indices] = np.clip(constrained[:, rate_indices], 0, 1)
    
    return constrained
