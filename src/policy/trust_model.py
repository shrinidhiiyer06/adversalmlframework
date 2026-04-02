import random
import numpy as np

def compute_device_trust(device_age_days, patch_compliance, anomaly_score):
    """
    Compute a sophisticated device trust score based on measurable posture attributes.
    
    Args:
        device_age_days: Number of days since last full hardware/OS verification
        patch_compliance: Float (0-1) representing OS/Security patching level
        anomaly_score: Float (0-1) representing behavioral anomaly detection score
        
    Returns:
        trust_score: Float (0-1) where 1.0 is maximum trust
        
    This model follows Zero-Trust principles by evaluating device posture
    rather than relying on static network location.
    """
    
    # 1. Age factor: Trust decays as the device hasn't been verified
    # Max trust for new/recently verified devices (0-30 days)
    # Minimum trust factor for devices older than 1 year
    age_factor = max(0.4, 1.0 - (device_age_days / 365.0))
    
    # 2. Compliance factor: Direct weight based on patch levels
    compliance_factor = patch_compliance
    
    # 3. Anomaly factor: Inversely proportional to behavioral risk
    behavior_factor = 1.0 - anomaly_score
    
    # Weighted calculation
    # Compliance is most important (0.4)
    # Behavior/Anomaly is critical (0.4)
    # Age/Verification history is a supporting factor (0.2)
    trust_score = (0.2 * age_factor +
                   0.4 * compliance_factor +
                   0.4 * behavior_factor)
    
    return float(np.clip(trust_score, 0.0, 1.0))

class TrustScoreGenerator:
    """Utility to generate realistic trust context for simulation"""
    
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        
    def generate_random_posture(self):
        """Generates random but realistic posture data for simulation"""
        return {
            'device_age_days': self.rng.randint(0, 400),
            'patch_compliance': self.rng.uniform(0.6, 1.0),
            'anomaly_score': self.rng.choice([0.05, 0.1, 0.3, 0.7], p=[0.7, 0.15, 0.1, 0.05])
        }
        
    def get_trust_score(self):
        posture = self.generate_random_posture()
        return compute_device_trust(**posture)
