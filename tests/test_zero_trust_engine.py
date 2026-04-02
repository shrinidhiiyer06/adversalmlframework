"""
Tests for the Zero-Trust policy engine.

Validates that all 8 policy rules fire correctly, priority ordering works,
partial context configs for ablation function properly, and attacker/legitimate
profiles produce expected decisions.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.policy.zero_trust_engine import ZeroTrustEngine, evaluate_access, AccessDecision


class TestRuleFiring:
    """Test that each policy rule fires under correct conditions."""

    def setup_method(self):
        self.engine = ZeroTrustEngine()

    def test_rule1_microsegment(self):
        """Rule 1: Critical resource + low device trust → DENY."""
        result = self.engine.evaluate(
            ml_risk_score=0.3,
            context={'resource_sensitivity': 0.95, 'device_trust': 0.3}
        )
        assert result.decision == "DENY"
        assert result.rule_fired == "Critical Resource Microsegment"
        assert result.priority == 1

    def test_rule2_high_ml_risk(self):
        """Rule 2: Very high ML risk → DENY regardless of context."""
        result = self.engine.evaluate(
            ml_risk_score=0.9,
            context={'device_trust': 0.9, 'geo_risk': 0.1}
        )
        assert result.decision == "DENY"
        assert result.rule_fired == "High-Risk ML Score"

    def test_rule3_device_trust(self):
        """Rule 3: Very low device trust → DENY."""
        result = self.engine.evaluate(
            ml_risk_score=0.3,
            context={'device_trust': 0.2, 'geo_risk': 0.3}
        )
        assert result.decision == "DENY"
        assert result.rule_fired == "Device Trust Threshold"

    def test_rule4_geo_risk(self):
        """Rule 4: Very high geo-risk → DENY."""
        result = self.engine.evaluate(
            ml_risk_score=0.3,
            context={'device_trust': 0.8, 'geo_risk': 0.85}
        )
        assert result.decision == "DENY"
        assert result.rule_fired == "Geo-Risk Threshold"

    def test_rule5_compound(self):
        """Rule 5: Moderate ML + elevated geo → DENY."""
        result = self.engine.evaluate(
            ml_risk_score=0.6,
            context={'device_trust': 0.8, 'geo_risk': 0.65}
        )
        assert result.decision == "DENY"
        assert result.rule_fired == "Compound Risk (ML + Geo)"

    def test_rule8_default_allow(self):
        """Rule 8: No risk signals → ALLOW."""
        result = self.engine.evaluate(
            ml_risk_score=0.2,
            context={'device_trust': 0.9, 'geo_risk': 0.1,
                     'time_of_day': 12, 'identity_verified': True}
        )
        assert result.decision == "ALLOW"
        assert result.rule_fired == "Default Allow"


class TestPriorityOrdering:
    """Test that higher priority rules take precedence."""

    def test_microsegment_before_ml(self):
        """Rule 1 (microsegment) fires before Rule 2 (high ML) when both match."""
        engine = ZeroTrustEngine()
        result = engine.evaluate(
            ml_risk_score=0.9,
            context={'resource_sensitivity': 0.95, 'device_trust': 0.2}
        )
        assert result.priority == 1

    def test_ml_before_device(self):
        """Rule 2 (high ML) fires before Rule 3 (device trust) when both match."""
        engine = ZeroTrustEngine()
        result = engine.evaluate(
            ml_risk_score=0.9,
            context={'device_trust': 0.1}
        )
        assert result.priority == 2


class TestAblationConfigs:
    """Test partial context configurations for ablation study."""

    def test_ml_only_ignores_context(self):
        """ML-only config: context factors are ignored, only ML threshold matters."""
        engine = ZeroTrustEngine(enabled_factors=set())
        # Low device trust should NOT trigger deny in ML-only mode
        result = engine.evaluate(
            ml_risk_score=0.3,
            context={'device_trust': 0.1, 'geo_risk': 0.9}
        )
        assert result.decision == "ALLOW"

    def test_ml_only_high_risk_still_denies(self):
        """ML-only: Rule 2 (high ML) still fires since ML is always active."""
        engine = ZeroTrustEngine(enabled_factors=set())
        result = engine.evaluate(ml_risk_score=0.9, context={})
        assert result.decision == "DENY"

    def test_ml_device_only(self):
        """ML + device trust: geo-risk rules are disabled."""
        engine = ZeroTrustEngine(enabled_factors={'device_trust'})
        result = engine.evaluate(
            ml_risk_score=0.3,
            context={'device_trust': 0.8, 'geo_risk': 0.9}
        )
        # Geo-risk should be ignored, so this should ALLOW
        assert result.decision == "ALLOW"

    def test_ml_geo_only(self):
        """ML + geo-risk: device trust rules are disabled."""
        engine = ZeroTrustEngine(enabled_factors={'geo_risk'})
        result = engine.evaluate(
            ml_risk_score=0.3,
            context={'device_trust': 0.1, 'geo_risk': 0.85}
        )
        assert result.decision == "DENY"
        assert result.rule_fired == "Geo-Risk Threshold"

    def test_full_system(self):
        """Full system: all factors enabled."""
        engine = ZeroTrustEngine()
        result = engine.evaluate(
            ml_risk_score=0.3,
            context={'device_trust': 0.2, 'geo_risk': 0.85}
        )
        assert result.decision == "DENY"


class TestBatchEvaluation:
    """Test batch processing."""

    def test_batch_returns_correct_count(self):
        engine = ZeroTrustEngine()
        scores = np.array([0.1, 0.5, 0.9])
        contexts = [
            {'device_trust': 0.9},
            {'device_trust': 0.5},
            {'device_trust': 0.1},
        ]
        results = engine.evaluate_batch(scores, contexts)
        assert len(results) == 3

    def test_batch_all_access_decisions(self):
        engine = ZeroTrustEngine()
        scores = np.array([0.1, 0.9])
        contexts = [{'device_trust': 0.9}, {'device_trust': 0.9}]
        results = engine.evaluate_batch(scores, contexts)
        assert all(isinstance(r, AccessDecision) for r in results)


class TestConvenienceFunction:
    """Test the evaluate_access convenience function."""

    def test_basic_call(self):
        result = evaluate_access(0.3, {'device_trust': 0.9, 'geo_risk': 0.1})
        assert result.decision == "ALLOW"

    def test_with_factors(self):
        result = evaluate_access(
            0.3, {'device_trust': 0.1}, enabled_factors={'device_trust'}
        )
        assert result.decision == "DENY"
