import unittest
import numpy as np
from src.policy.zero_trust_engine import ZeroTrustPolicyEngine, AccessDecision
from src.policy.network_context import NetworkRequestContext


class TestZeroTrustPolicyEngine(unittest.TestCase):
    def setUp(self):
        self.engine = ZeroTrustPolicyEngine()
        # Create a mock context
        self.mock_features = np.zeros(41)
        self.context = NetworkRequestContext(
            flow_id="test_flow",
            user_identity="test_user",
            device_trust_score=0.9,
            device_posture={'patch_compliance': 1.0, 'anomaly_score': 0.0},
            geo_risk_score=0.1,
            time_of_day_risk=0.1,
            source_ip="1.1.1.1",
            dest_ip="10.0.0.1",
            requested_segment="web",
            flow_features=self.mock_features
        )

    def test_allow_decision(self):
        """Test standard ALLOW case"""
        decision, reason = self.engine.evaluate_access(ml_risk_score=0.1, context=self.context)
        self.assertEqual(decision, AccessDecision.ALLOW)
        self.assertIn("All checks passed", reason)

    def test_high_ml_risk_deny(self):
        """Test DENY due to high ML risk"""
        decision, reason = self.engine.evaluate_access(ml_risk_score=0.9, context=self.context)
        self.assertEqual(decision, AccessDecision.DENY)
        # Sensitive resource check happens first
        self.assertIn("exceeds threshold for sensitive resource 'web'", reason)

    def test_low_device_trust_mfa(self):
        """Test STEP_UP_AUTH due to low device trust"""
        self.context.device_trust_score = 0.3
        decision, reason = self.engine.evaluate_access(ml_risk_score=0.2, context=self.context)
        self.assertEqual(decision, AccessDecision.STEP_UP_AUTH)
        self.assertIn("Untrusted device requires MFA", reason)

    def test_low_device_trust_elevated_risk_deny(self):
        """Test DENY due to low device trust + moderate risk"""
        self.context.device_trust_score = 0.3
        decision, reason = self.engine.evaluate_access(ml_risk_score=0.5, context=self.context)
        self.assertEqual(decision, AccessDecision.DENY)
        self.assertIn("Untrusted device + elevated ML risk", reason)

    def test_high_geo_risk_mfa(self):
        """Test STEP_UP_AUTH due to high geo risk"""
        self.context.geo_risk_score = 0.8
        decision, reason = self.engine.evaluate_access(ml_risk_score=0.2, context=self.context)
        self.assertEqual(decision, AccessDecision.STEP_UP_AUTH)
        self.assertIn("High geo risk", reason)

    def test_sensitive_segment_deny_on_low_risk(self):
        """Test micro-segmentation DENY if ML risk exceeds segment-specific threshold"""
        self.context.requested_segment = "admin" # threshold 0.4
        decision, reason = self.engine.evaluate_access(ml_risk_score=0.45, context=self.context)
        self.assertEqual(decision, AccessDecision.DENY)
        self.assertIn("exceeds threshold for sensitive resource 'admin'", reason)

    def test_rate_limit(self):
        """Test RATE_LIMIT for moderate ML risk"""
        decision, reason = self.engine.evaluate_access(ml_risk_score=0.7, context=self.context)
        self.assertEqual(decision, AccessDecision.RATE_LIMIT)
        self.assertIn("Moderate risk", reason)


if __name__ == '__main__':
    unittest.main()
