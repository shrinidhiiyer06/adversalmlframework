"""
Network Evasion Scenarios
Realistic network evasion attack patterns
"""

import numpy as np


class NetworkEvasionScenarios:
    """Realistic network evasion attack scenarios"""
    
    @staticmethod
    def slow_rate_limit_evasion(flow_features):
        """
        Attacker slows down attack rate to evade detection
        - Reduces packet rate
        - Spreads attack over time
        
        Args:
            flow_features: Network flow feature vector
            
        Returns:
            Modified flow features
        """
        modified = flow_features.copy()
        
        # Reduce srv_count (service access rate) - feature index 23
        if len(modified) > 23:
            modified[23] = max(1, modified[23] * 0.5)
        
        # Increase duration - feature index 0
        if len(modified) > 0:
            modified[0] = modified[0] * 2
        
        return modified
    
    @staticmethod
    def port_hopping_evasion(flow_features):
        """
        Attacker changes ports to avoid pattern detection
        - Modifies service diversity metrics
        
        Args:
            flow_features: Network flow feature vector
            
        Returns:
            Modified flow features
        """
        modified = flow_features.copy()
        
        # Increase same_srv_rate (looks more normal) - feature index 28
        if len(modified) > 28:
            modified[28] = min(1.0, modified[28] + 0.1)
        
        # Decrease diff_srv_rate - feature index 29
        if len(modified) > 29:
            modified[29] = max(0.0, modified[29] - 0.1)
        
        return modified
    
    @staticmethod
    def mimicry_attack(flow_features, benign_template):
        """
        Attacker mimics benign traffic patterns
        
        Args:
            flow_features: Malicious flow features
            benign_template: Benign flow to mimic
            
        Returns:
            Blended flow features
        """
        # Blend malicious flow with benign template
        alpha = 0.7  # 70% benign features
        modified = alpha * benign_template + (1 - alpha) * flow_features
        return modified
    
    @staticmethod
    def fragmentation_evasion(flow_features):
        """
        Attacker fragments packets to evade detection
        - Reduces packet sizes
        - Increases packet count
        
        Args:
            flow_features: Network flow feature vector
            
        Returns:
            Modified flow features
        """
        modified = flow_features.copy()
        
        # Reduce src_bytes and dst_bytes - features 4 and 5
        if len(modified) > 5:
            modified[4] = modified[4] * 0.7  # src_bytes
            modified[5] = modified[5] * 0.7  # dst_bytes
        
        # Increase count (number of connections) - feature 22
        if len(modified) > 22:
            modified[22] = modified[22] * 1.3
        
        return modified
