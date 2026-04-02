import unittest
import numpy as np
import torch
import torch.nn as nn
from src.attacks.network_adversarial import NetworkAdversarialAttacker


class MockModel(nn.Module):
    def __init__(self):
        super(MockModel, self).__init__()
        self.fc = nn.Linear(41, 1)
        # Set weights to be sensitive to the 21st feature (index 20, non-integer)
        with torch.no_grad():
            self.fc.weight.fill_(0.0)
            self.fc.weight[0, 20] = 1.0
            self.fc.bias.fill_(0.0)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))


class TestNetworkAdversarialAttacker(unittest.TestCase):
    def setUp(self):
        self.model = MockModel()
        self.bounds = [(0.0, 1.0)] * 41
        self.feature_names = ["feature_" + str(i) for i in range(41)]
        self.attacker = NetworkAdversarialAttacker(self.model, self.bounds, self.feature_names)
        self.x_clean = np.zeros(41)
        self.x_clean[20] = 0.8  # High value -> high risk in MockModel

    def test_fgsm_attack_generation(self):
        """Test that FGSM generates a perturbed sample"""
        # epsilon=0.1, we want to decrease risk (target_label=0)
        x_adv = self.attacker.constrained_fgsm(self.x_clean, epsilon=0.1, target_label=0)
        
        self.assertEqual(x_adv.shape, (1, 41))
        # Feature should have decreased
        self.assertLess(x_adv[0, 20], self.x_clean[20])
        # We allow for small rounding/safety margin in constraints
        self.assertLess(x_adv[0, 20], 0.75)

    def test_attack_constraints(self):
        """Test that attacks respect feature bounds"""
        # Set feature near lower bound
        self.x_clean[20] = 0.05
        x_adv = self.attacker.constrained_fgsm(self.x_clean, epsilon=0.1, target_label=0)
        
        # Should be clipped to 0.0 (or domain min)
        self.assertGreaterEqual(x_adv[0, 20], 0.0)
        self.assertTrue(np.all(x_adv >= -2.0)) # Scaled safety floor
        self.assertTrue(np.all(x_adv <= 1.0))

    def test_attack_success_evaluation(self):
        """Test attack success rate calculation with new nested metrics"""
        X = np.array([self.x_clean])
        results = self.attacker.evaluate_attack(X, epsilon=0.1, threshold=0.5)
        
        self.assertIn('clean_metrics', results)
        self.assertIn('adv_metrics', results)
        self.assertIn('attack_success_rate', results)
        self.assertIn('roc_auc', results['adv_metrics'])


if __name__ == '__main__':
    unittest.main()
