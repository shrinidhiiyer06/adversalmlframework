"""
Tests for the network constraint validation module.

Validates that the constraint checker correctly catches negative features,
non-integer integer features, and out-of-bound values.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.attacks.validate_constraints import (
    validate_single_sample,
    validate_adversarial_batch,
)


class TestSingleSample:
    """Test individual sample validation."""

    def test_valid_sample(self):
        """A perfectly valid sample should pass all checks."""
        sample = np.array([500.0, 1.5, 3.0, 7.2, 5.0, 80.0])
        result = validate_single_sample(sample)
        assert result['valid'] is True
        assert len(result['violations']) == 0

    def test_negative_value_caught(self):
        """Negative feature values must be flagged."""
        sample = np.array([-10.0, 1.5, 3.0, 7.2, 5.0, 80.0])
        result = validate_single_sample(sample)
        assert result['valid'] is False
        assert any('negative' in v or 'below minimum' in v
                    for v in result['violations'])

    def test_non_integer_caught(self):
        """Integer features with non-integer values must be flagged."""
        sample = np.array([500.7, 1.5, 3.0, 7.2, 5.0, 80.0])
        result = validate_single_sample(sample)
        assert result['valid'] is False

    def test_out_of_bounds_caught(self):
        """Values exceeding bounds must be flagged."""
        sample = np.array([500.0, 1.5, 3.0, 9.0, 5.0, 80.0])  # entropy > 8
        result = validate_single_sample(sample)
        assert result['valid'] is False


class TestBatchValidation:
    """Test batch validation with auto-fix."""

    def test_all_valid(self):
        """Batch of valid samples should have 100% pass rate."""
        X = np.array([
            [500, 1.5, 3, 7.2, 5, 80],
            [300, 2.0, 5, 6.5, 10, 90],
        ])
        result = validate_adversarial_batch(X)
        assert result['pass_rate'] == 1.0

    def test_mixed_validity(self):
        """Batch with one invalid should have 50% pass rate."""
        X = np.array([
            [500, 1.5, 3, 7.2, 5, 80],     # valid
            [-10, 1.5, 3, 7.2, 5, 80],     # invalid (negative)
        ])
        result = validate_adversarial_batch(X)
        assert result['pass_rate'] == 0.5

    def test_fix_violations(self):
        """Auto-fix should clip values to valid ranges."""
        X = np.array([
            [-10, -1.0, -5, 9.0, -3, 150],  # all bad
        ])
        result = validate_adversarial_batch(X, fix_violations=True)
        assert 'X_fixed' in result
        X_fixed = result['X_fixed']
        assert X_fixed[0, 0] >= 64   # packet_size minimum
        assert X_fixed[0, 3] <= 8.0  # entropy max
        assert X_fixed[0, 5] <= 100  # trust_score max
        assert result['pass_rate_after_fix'] == 1.0
