"""
Network Context Builder
Generates Zero-Trust context for network flows
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
import hashlib


from src.policy.trust_model import compute_device_trust, TrustScoreGenerator

@dataclass
class NetworkRequestContext:
    """Zero-Trust context for each network flow"""
    flow_id: str
    user_identity: str
    device_trust_score: float  # 0-1
    device_posture: dict       # Raw posture data
    geo_risk_score: float      # 0-1  
    time_of_day_risk: float    # 0-1
    source_ip: str
    dest_ip: str
    requested_segment: str     # e.g., "database", "web", "admin"
    flow_features: np.ndarray


class NetworkContextBuilder:
    """Generates realistic Zero-Trust context for network flows"""
    
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        self.trust_gen = TrustScoreGenerator(seed=seed)
        
    def build_context(self, flow_features, flow_index):
        """
        Simulate Zero-Trust metadata for a network flow
        
        Sourcing from trust_model.py logic for realistic security posture.
        """
        # Generate deterministic user ID from flow
        user_id = f"user_{hashlib.md5(str(flow_index).encode()).hexdigest()[:8]}"
        
        # Simulate realistic device posture and compute logic-driven trust
        posture = self.trust_gen.generate_random_posture()
        device_trust = compute_device_trust(**posture)
        
        # Simulate geo-risk (some IPs from risky regions)
        geo_risk = self.rng.choice([0.1, 0.3, 0.7], p=[0.7, 0.2, 0.1])
        
        # Time-based risk (higher risk at unusual hours)
        time_risk = self.rng.uniform(0.1, 0.5)
        
        # Simulate network segment access
        segments = ["web", "database", "admin", "api", "internal"]
        segment = self.rng.choice(segments)
        
        # Generate IPs
        src_ip = f"{self.rng.randint(1, 255)}.{self.rng.randint(1, 255)}.{self.rng.randint(1, 255)}.{self.rng.randint(1, 255)}"
        dst_ip = f"10.0.{self.rng.randint(1, 255)}.{self.rng.randint(1, 255)}"
        
        return NetworkRequestContext(
            flow_id=f"flow_{flow_index}",
            user_identity=user_id,
            device_trust_score=device_trust,
            device_posture=posture,
            geo_risk_score=geo_risk,
            time_of_day_risk=time_risk,
            source_ip=src_ip,
            dest_ip=dst_ip,
            requested_segment=segment,
            flow_features=flow_features
        )
