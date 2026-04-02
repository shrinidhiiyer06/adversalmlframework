"""
Configuration module with validation and best practices.

All configuration values are validated on import to catch issues early.
Uses environment variables for sensitive/deployment-specific values.
"""

import os
from pathlib import Path
from typing import List
import logging

logger = logging.getLogger(__name__)

# ==================== 1. PATHS ====================

# Determine base directory (where this config.py file is located)
# This makes paths work regardless of where the script is run from
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Model and data directories
MODEL_DIR = os.getenv('MODEL_DIR', str(PROJECT_ROOT / "models"))
DATA_DIR = os.getenv('DATA_DIR', str(PROJECT_ROOT / "data"))
RESULTS_DIR = os.getenv('RESULTS_DIR', str(PROJECT_ROOT / "results"))
SCRIPTS_DIR = os.getenv('SCRIPTS_DIR', str(PROJECT_ROOT / "scripts"))
FIGURES_DIR = os.getenv('FIGURES_DIR', str(PROJECT_ROOT / "figures"))
CICIDS_DIR = os.getenv('CICIDS_DIR', str(PROJECT_ROOT / "data" / "cicids2017"))

# Demo sample file for reproducible Research Demo tab
DEMO_SAMPLES_PATH = os.getenv('DEMO_SAMPLES_PATH', str(PROJECT_ROOT / "data" / "demo_samples.npy"))

# Create directories if they don't exist (safe initialization)
for directory in [MODEL_DIR, DATA_DIR, RESULTS_DIR, SCRIPTS_DIR, FIGURES_DIR]:
    Path(directory).mkdir(parents=True, exist_ok=True)

# ==================== 2. GLOBAL SETTINGS ====================

# Random seed for reproducibility
RANDOM_SEED = int(os.getenv('RANDOM_SEED', '42'))

# Multi-seed values for statistically rigorous evaluation
# Every metric must be reported as mean ± std across these 5 seeds
MULTI_SEED_VALUES: List[int] = [0, 42, 123, 456, 789]

# Debug mode (set to False in production)
DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() in ('true', '1', 'yes')

# Logging level
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()

# ==================== 3. MODEL PERFORMANCE THRESHOLDS ====================

# Minimum accuracy required for CI/CD to pass
CI_ACCURACY_THRESHOLD = float(os.getenv('CI_ACCURACY_THRESHOLD', '0.80'))

# Minimum accuracy for production deployment
PRODUCTION_ACCURACY_THRESHOLD = float(os.getenv('PRODUCTION_ACCURACY_THRESHOLD', '0.85'))

# ==================== 4. ATTACK CONFIGURATIONS ====================

# Epsilon values for robustness curve sweeps (research publication range)
# Focused on realistic threat region where adversarial examples remain valid
EPS_VALUES: List[float] = [0.01, 0.02, 0.05, 0.10, 0.20]

# HopSkipJump (Black-box) attack parameters
HSJ_MAX_ITER = int(os.getenv('HSJ_MAX_ITER', '50'))
HSJ_MAX_EVAL = int(os.getenv('HSJ_MAX_EVAL', '100'))
HSJ_INIT_EVAL = int(os.getenv('HSJ_INIT_EVAL', '10'))

# Fast Gradient Method (White-box) attack parameters
FGM_EPS = float(os.getenv('FGM_EPS', '0.2'))

# PGD (Projected Gradient Descent) attack parameters — formally documented
# for paper methodology section reproducibility
PGD_ITERATIONS = int(os.getenv('PGD_ITERATIONS', '40'))
PGD_ALPHA_FACTOR = int(os.getenv('PGD_ALPHA_FACTOR', '10'))  # alpha = eps / PGD_ALPHA_FACTOR
PGD_RESTARTS = int(os.getenv('PGD_RESTARTS', '1'))

# Default sample sizes for attacks
DEFAULT_ATTACK_SAMPLE_SIZE = int(os.getenv('ATTACK_SAMPLE_SIZE', '100'))

# Demo samples for Research Demo tab (30 high-confidence malicious samples)
DEMO_SAMPLE_COUNT = 30
DEMO_MIN_RISK_SCORE = 0.7

# ==================== 5. TRAINING CONFIGURATIONS ====================

# Surrogate model training parameters
SURROGATE_LR = float(os.getenv('SURROGATE_LR', '0.001'))
SURROGATE_EPOCHS = int(os.getenv('SURROGATE_EPOCHS', '100'))
EARLY_STOPPING_PATIENCE = int(os.getenv('EARLY_STOPPING_PATIENCE', '5'))
TRAIN_VAL_SPLIT = float(os.getenv('TRAIN_VAL_SPLIT', '0.2'))

# Batch size for training
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '128'))

# ==================== 6. ENSEMBLE DEFENSE SETTINGS ====================

# Confidence threshold for uncertainty rejection
# Traffic with prob_attack in [0.5 - threshold, 0.5 + threshold] is rejected
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.15'))

# Isolation Forest contamination rate (expected fraction of anomalies)
ISOLATION_CONTAMINATION = float(os.getenv('ISOLATION_CONTAMINATION', '0.05'))

# ==================== 7. STREAMLIT APP SETTINGS ====================

# Port for Streamlit app
STREAMLIT_PORT = int(os.getenv('STREAMLIT_PORT', '8501'))

# Enable/disable certain features
ENABLE_RED_TEAM_TAB = os.getenv('ENABLE_RED_TEAM', 'True').lower() in ('true', '1', 'yes')
ENABLE_BLUE_TEAM_TAB = os.getenv('ENABLE_BLUE_TEAM', 'True').lower() in ('true', '1', 'yes')

# ==================== 8. PERFORMANCE SETTINGS ====================

# Maximum number of samples to process at once (prevent OOM)
MAX_BATCH_SIZE = int(os.getenv('MAX_BATCH_SIZE', '10000'))

# Timeout for long operations (seconds)
OPERATION_TIMEOUT = int(os.getenv('OPERATION_TIMEOUT', '300'))

# ==================== VALIDATION ====================

def validate_config() -> None:
    """
    Validate all configuration values.
    
    Raises:
        ValueError: If any configuration value is invalid
    """
    errors = []
    
    # Validate numeric ranges
    if not 0 < CI_ACCURACY_THRESHOLD <= 1.0:
        errors.append(f"CI_ACCURACY_THRESHOLD must be in (0, 1], got {CI_ACCURACY_THRESHOLD}")
    
    if not 0 < PRODUCTION_ACCURACY_THRESHOLD <= 1.0:
        errors.append(f"PRODUCTION_ACCURACY_THRESHOLD must be in (0, 1], got {PRODUCTION_ACCURACY_THRESHOLD}")
    
    if not 0 < CONFIDENCE_THRESHOLD < 0.5:
        errors.append(f"CONFIDENCE_THRESHOLD must be in (0, 0.5), got {CONFIDENCE_THRESHOLD}")
    
    if not 0 < ISOLATION_CONTAMINATION < 1.0:
        errors.append(f"ISOLATION_CONTAMINATION must be in (0, 1), got {ISOLATION_CONTAMINATION}")
    
    if not 0 < TRAIN_VAL_SPLIT < 1.0:
        errors.append(f"TRAIN_VAL_SPLIT must be in (0, 1), got {TRAIN_VAL_SPLIT}")
    
    if SURROGATE_LR <= 0:
        errors.append(f"SURROGATE_LR must be positive, got {SURROGATE_LR}")
    
    if SURROGATE_EPOCHS < 1:
        errors.append(f"SURROGATE_EPOCHS must be >= 1, got {SURROGATE_EPOCHS}")
    
    if EARLY_STOPPING_PATIENCE < 1:
        errors.append(f"EARLY_STOPPING_PATIENCE must be >= 1, got {EARLY_STOPPING_PATIENCE}")
    
    if FGM_EPS < 0:
        errors.append(f"FGM_EPS must be non-negative, got {FGM_EPS}")
    
    if HSJ_MAX_ITER < 1:
        errors.append(f"HSJ_MAX_ITER must be >= 1, got {HSJ_MAX_ITER}")
    
    if BATCH_SIZE < 1:
        errors.append(f"BATCH_SIZE must be >= 1, got {BATCH_SIZE}")
    
    if MAX_BATCH_SIZE < BATCH_SIZE:
        errors.append(f"MAX_BATCH_SIZE ({MAX_BATCH_SIZE}) must be >= BATCH_SIZE ({BATCH_SIZE})")
    
    # Validate epsilon values
    if not EPS_VALUES:
        errors.append("EPS_VALUES cannot be empty")
    
    if any(eps < 0 for eps in EPS_VALUES):
        errors.append(f"All EPS_VALUES must be non-negative, got {EPS_VALUES}")
    
    if EPS_VALUES != sorted(EPS_VALUES):
        logger.warning("EPS_VALUES are not sorted, sorting automatically")
        EPS_VALUES.sort()
    
    # Validate paths exist
    for name, path in [('MODEL_DIR', MODEL_DIR), ('DATA_DIR', DATA_DIR), ('RESULTS_DIR', RESULTS_DIR)]:
        if not os.path.exists(path):
            logger.warning(f"{name} does not exist: {path} (will be created)")
    
    # Raise if any errors
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {err}" for err in errors)
        raise ValueError(error_msg)
    
    logger.debug("Configuration validation passed")


# ==================== EXPORT SUMMARY ====================

def get_config_summary() -> dict:
    """
    Get a summary of all configuration values.
    
    Returns:
        Dictionary with all config values
    """
    return {
        'paths': {
            'BASE_DIR': str(BASE_DIR),
            'PROJECT_ROOT': str(PROJECT_ROOT),
            'MODEL_DIR': MODEL_DIR,
            'DATA_DIR': DATA_DIR,
            'RESULTS_DIR': RESULTS_DIR
        },
        'global_settings': {
            'RANDOM_SEED': RANDOM_SEED,
            'DEBUG_MODE': DEBUG_MODE,
            'LOG_LEVEL': LOG_LEVEL
        },
        'thresholds': {
            'CI_ACCURACY_THRESHOLD': CI_ACCURACY_THRESHOLD,
            'PRODUCTION_ACCURACY_THRESHOLD': PRODUCTION_ACCURACY_THRESHOLD,
            'CONFIDENCE_THRESHOLD': CONFIDENCE_THRESHOLD
        },
        'attack_params': {
            'EPS_VALUES': EPS_VALUES,
            'HSJ_MAX_ITER': HSJ_MAX_ITER,
            'HSJ_MAX_EVAL': HSJ_MAX_EVAL,
            'FGM_EPS': FGM_EPS
        },
        'training_params': {
            'SURROGATE_LR': SURROGATE_LR,
            'SURROGATE_EPOCHS': SURROGATE_EPOCHS,
            'BATCH_SIZE': BATCH_SIZE
        }
    }


# ==================== AUTO-VALIDATE ON IMPORT ====================

try:
    validate_config()
    if DEBUG_MODE:
        logger.debug("Configuration loaded and validated successfully")
except ValueError as e:
    logger.error(f"Configuration validation failed: {e}")
    raise


# ==================== BACKWARDS COMPATIBILITY ====================
# Old code might expect these exact names, so we keep them as aliases

# For backwards compatibility with old import style
if __name__ != '__main__':
    import sys
    sys.modules[__name__].BASE_DIR = str(BASE_DIR)
    sys.modules[__name__].PROJECT_ROOT = str(PROJECT_ROOT)
