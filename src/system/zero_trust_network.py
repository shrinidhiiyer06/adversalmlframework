"""
Zero-Trust Network System
Complete integration of all components
"""

import torch
import numpy as np
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.risk_engine.network_classifier import NetworkRiskClassifier
from src.policy.network_context import NetworkContextBuilder
from src.policy.zero_trust_engine import ZeroTrustPolicyEngine, AccessDecision


class ZeroTrustNetworkSystem:
    """
    Complete Zero-Trust network access control system
    
    Pipeline:
    Network Flow → Feature Extraction → ML Risk Scoring → 
    Context Enrichment → Policy Evaluation → Access Decision
    """
    
    def __init__(self, model_path):
        """
        Initialize Zero-Trust network system
        
        Args:
            model_path: Path to trained NetworkRiskClassifier model
        """
        # Load ML risk classifier
        self.risk_model = NetworkRiskClassifier()
        
        if os.path.exists(model_path):
            self.risk_model.load_state_dict(torch.load(model_path))
            print(f"Loaded risk model from {model_path}")
        else:
            print(f"Warning: Model not found at {model_path}, using untrained model")
        
        self.risk_model.eval()
        
        # Initialize components
        self.context_builder = NetworkContextBuilder()
        self.policy_engine = ZeroTrustPolicyEngine()
        
        # Telemetry
        self.access_log = []
        
    def process_network_request(self, flow_features, flow_index):
        """
        Process a network access request through Zero-Trust pipeline
        
        Args:
            flow_features: Network flow feature vector
            flow_index: Index of the flow
            
        Returns:
            Dictionary with decision, reason, scores, and context
        """
        # Step 1: ML Risk Scoring
        with torch.no_grad():
            if isinstance(flow_features, np.ndarray):
                flow_tensor = torch.FloatTensor(flow_features)
            else:
                flow_tensor = flow_features
            
            if len(flow_tensor.shape) == 1:
                flow_tensor = flow_tensor.unsqueeze(0)
            
            risk_score = self.risk_model(flow_tensor).item()
        
        # Step 2: Build Zero-Trust context
        context = self.context_builder.build_context(flow_features, flow_index)
        
        # Step 3: Policy evaluation
        decision, reason = self.policy_engine.evaluate_access(
            ml_risk_score=risk_score,
            context=context
        )
        
        # Step 4: Log decision
        log_entry = self.policy_engine.log_decision(
            decision, reason, context, risk_score
        )
        self.access_log.append(log_entry)
        
        return {
            'decision': decision,
            'reason': reason,
            'ml_risk_score': float(risk_score),
            'context': context,
            'log_entry': log_entry
        }
    
    def evaluate_adversarial_evasion(self, X_clean, X_adv, flow_indices):
        """
        Test if adversarial attacks can bypass Zero-Trust controls
        
        Args:
            X_clean: Clean malicious samples
            X_adv: Adversarial samples
            flow_indices: Indices for flows
            
        Returns:
            Dictionary with evasion metrics
        """
        results = {
            'clean': [],
            'adversarial': []
        }
        
        print(f"Evaluating {len(X_clean)} clean samples...")
        # Process clean samples
        for i, (x, idx) in enumerate(zip(X_clean, flow_indices)):
            result = self.process_network_request(x, idx)
            results['clean'].append(result)
        
        print(f"Evaluating {len(X_adv)} adversarial samples...")
        # Process adversarial samples
        for i, (x, idx) in enumerate(zip(X_adv, flow_indices)):
            result = self.process_network_request(x, idx)
            results['adversarial'].append(result)
        
        # Calculate evasion metrics
        clean_denies = sum(1 for r in results['clean'] if r['decision'] == AccessDecision.DENY)
        adv_allows = sum(1 for r in results['adversarial'] if r['decision'] == AccessDecision.ALLOW)
        adv_denies = sum(1 for r in results['adversarial'] if r['decision'] == AccessDecision.DENY)
        
        evasion_rate = adv_allows / len(X_adv) if len(X_adv) > 0 else 0
        
        # Calculate average risk score change
        clean_risks = [r['ml_risk_score'] for r in results['clean']]
        adv_risks = [r['ml_risk_score'] for r in results['adversarial']]
        
        return {
            'evasion_success_rate': evasion_rate,
            'clean_deny_rate': clean_denies / len(X_clean),
            'adv_allow_rate': adv_allows / len(X_adv),
            'adv_deny_rate': adv_denies / len(X_adv),
            'avg_clean_risk': np.mean(clean_risks),
            'avg_adv_risk': np.mean(adv_risks),
            'risk_reduction': np.mean(clean_risks) - np.mean(adv_risks),
            'detailed_results': results
        }
    
    def get_decision_summary(self):
        """Get summary of all access decisions"""
        if not self.access_log:
            return "No decisions logged yet"
        
        decisions = [log['decision'] for log in self.access_log]
        summary = {
            'total': len(decisions),
            'ALLOW': decisions.count('ALLOW'),
            'DENY': decisions.count('DENY'),
            'STEP_UP_AUTH': decisions.count('STEP_UP_AUTH'),
            'RATE_LIMIT': decisions.count('RATE_LIMIT'),
            'ISOLATE': decisions.count('ISOLATE')
        }
        return summary
    
    def export_telemetry(self, filepath):
        """Export access logs to file"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.access_log, f, indent=2)
        print(f"Exported {len(self.access_log)} log entries to {filepath}")
