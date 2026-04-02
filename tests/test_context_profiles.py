"""
Tests for context profile generators.

Validates that attacker and legitimate profiles produce values within
the documented distributional ranges, and that the statistical properties
match the specified means and standard deviations.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.simulation.context_profiles import (
    generate_attacker_context,
    generate_legitimate_context,
    generate_mixed_contexts,
)


class TestAttackerProfiles:
    """Verify attacker context values fall within specified ranges."""

    def test_device_trust_range(self):
        """Attacker device trust must be in [0.1, 0.55]."""
        contexts = generate_attacker_context(100, seed=42)
        trusts = [c['device_trust'] for c in contexts]
        assert all(0.1 <= t <= 0.55 for t in trusts)

    def test_geo_risk_range(self):
        """Attacker geo-risk must be in [0.55, 0.95]."""
        contexts = generate_attacker_context(100, seed=42)
        risks = [c['geo_risk'] for c in contexts]
        assert all(0.55 <= r <= 0.95 for r in risks)

    def test_approximate_mean_device_trust(self):
        """Attacker device trust should be approximately μ=0.35."""
        contexts = generate_attacker_context(1000, seed=42)
        mean_trust = np.mean([c['device_trust'] for c in contexts])
        assert abs(mean_trust - 0.35) < 0.05  # Within 0.05 of target mean

    def test_approximate_mean_geo_risk(self):
        """Attacker geo-risk should be approximately μ=0.72."""
        contexts = generate_attacker_context(1000, seed=42)
        mean_risk = np.mean([c['geo_risk'] for c in contexts])
        assert abs(mean_risk - 0.72) < 0.05

    def test_identity_mostly_unverified(self):
        """Attackers should have low identity verification rate (~30%)."""
        contexts = generate_attacker_context(1000, seed=42)
        verified_rate = np.mean([c['identity_verified'] for c in contexts])
        assert verified_rate < 0.5


class TestLegitimateProfiles:
    """Verify legitimate context values fall within specified ranges."""

    def test_device_trust_range(self):
        """Legitimate device trust must be in [0.65, 1.0]."""
        contexts = generate_legitimate_context(100, seed=42)
        trusts = [c['device_trust'] for c in contexts]
        assert all(0.65 <= t <= 1.0 for t in trusts)

    def test_geo_risk_range(self):
        """Legitimate geo-risk must be in [0.05, 0.45]."""
        contexts = generate_legitimate_context(100, seed=42)
        risks = [c['geo_risk'] for c in contexts]
        assert all(0.05 <= r <= 0.45 for r in risks)

    def test_identity_mostly_verified(self):
        """Legitimate users should have high verification rate (~90%)."""
        contexts = generate_legitimate_context(1000, seed=42)
        verified_rate = np.mean([c['identity_verified'] for c in contexts])
        assert verified_rate > 0.7


class TestMixedContexts:
    """Verify mixed context generation."""

    def test_correct_counts(self):
        contexts, labels = generate_mixed_contexts(10, 20, seed=42)
        assert len(contexts) == 30
        assert sum(labels) == 10

    def test_shuffled(self):
        """Labels should be shuffled, not all attacks first."""
        contexts, labels = generate_mixed_contexts(10, 20, seed=42)
        # Extremely unlikely all attacks are at start after shuffle
        assert labels[:10] != [1] * 10 or labels[10:] != [0] * 20

    def test_reproducibility(self):
        """Same seed produces same results."""
        ctx1, lab1 = generate_mixed_contexts(5, 5, seed=123)
        ctx2, lab2 = generate_mixed_contexts(5, 5, seed=123)
        assert lab1 == lab2
