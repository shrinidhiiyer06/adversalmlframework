"""
Network Adversarial Attacks
Adversarial attacks adapted for network flow features
"""

import torch
import numpy as np
from src.traffic.feature_constraints import FEATURE_BOUNDS, CATEGORICAL_INDICES, apply_domain_constraints
from src.evaluation.network_metrics import evaluate_network_performance


class NetworkAdversarialAttacker:
    """Adversarial attacks adapted for network flow features"""
    
    def __init__(self, model, feature_bounds, feature_names=None):
        """
        Args:
            model: NetworkRiskClassifier
            feature_bounds: dict with 'min' and 'max' arrays or list of (min, max) tuples
            feature_names: List of feature names for domain-specific clipping
        """
        self.model = model
        self.feature_names = feature_names
        
        # Normalize bounds to dict for np.clip
        if isinstance(feature_bounds, list):
            self.bounds = {
                'min': np.array([b[0] for b in feature_bounds]),
                'max': np.array([b[1] for b in feature_bounds])
            }
        else:
            self.bounds = feature_bounds
        
        # Categorical features that should NOT receive gradients (e.g., protocols)
        self.categorical_indices = CATEGORICAL_INDICES
        # Integer features that must be rounded
        self.integer_features = [0, 4, 5, 22, 23, 31, 32]
        
    def constrained_fgsm(self, x, epsilon=0.01, target_label=0):
        """
        FGSM attack with network feature constraints
        
        Network features must stay realistic:
        - Duration must be positive
        - Byte counts must be integers
        - Flags must be valid
        
        Args:
            x: Input network flow features (numpy array)
            epsilon: Perturbation magnitude
            target_label: Target class (0=benign, 1=malicious)
            
        Returns:
            Adversarial network flow
        """
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        x_tensor = torch.FloatTensor(x).requires_grad_(True)
        
        # Forward pass
        self.model.eval()
        output = self.model(x_tensor)
        
        # Calculate loss
        target = torch.FloatTensor([[target_label]])
        loss = torch.nn.BCELoss()(output, target)
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # Generate perturbation
        gradients = x_tensor.grad.data
        
        # MASK GRADIENTS for categorical features to prevent invalid drift
        for idx in self.categorical_indices:
            if idx < gradients.shape[1]:
                gradients[:, idx] = 0
                
        perturbation = epsilon * gradients.sign()
        
        # Apply perturbation (minimize loss to evade detection)
        x_adv = x_tensor - perturbation
        
        # Apply feature-specific constraints
        x_adv = self._apply_network_constraints(x_adv.detach().numpy())
        
        return x_adv
    
    def pgd_attack(self, x, epsilon=0.05, alpha=0.01, num_iter=10, target_label=0):
        """
        Projected Gradient Descent attack for network flows
        
        Args:
            x: Input network flow features
            epsilon: Maximum perturbation
            alpha: Step size
            num_iter: Number of iterations
            target_label: Target class
            
        Returns:
            Adversarial network flow
        """
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        x_orig = x.copy()
        x_adv = x.copy()
        
        for i in range(num_iter):
            x_tensor = torch.FloatTensor(x_adv).requires_grad_(True)
            
            # Forward pass
            self.model.eval()
            output = self.model(x_tensor)
            
            # Calculate loss
            target = torch.FloatTensor([[target_label]])
            loss = torch.nn.BCELoss()(output, target)
            
            # Backward pass
            self.model.zero_grad()
            loss.backward()
            
            # Mask gradients for categorical features
            gradients = x_tensor.grad.data
            for idx in self.categorical_indices:
                if idx < gradients.shape[1]:
                    gradients[:, idx] = 0
            
            # Update adversarial example
            perturbation = alpha * gradients.sign()
            x_adv = x_tensor.detach().numpy() - perturbation.numpy()
            
            # Project back to epsilon ball
            delta = x_adv - x_orig
            delta = np.clip(delta, -epsilon, epsilon)
            x_adv = x_orig + delta
            
            # Apply constraints
            x_adv = self._apply_network_constraints(x_adv)
        
        return x_adv
    
    def _apply_network_constraints(self, x_adv):
        """Ensure adversarial samples remain valid network flows"""
        x_constrained = x_adv.copy()
        
        # 1. Clip to statistical feature ranges from training data
        x_constrained = np.clip(x_constrained, self.bounds['min'], self.bounds['max'])
        
        # 2. Apply domain-specific constraints (duration >= 0, etc.)
        if self.feature_names:
            x_constrained = apply_domain_constraints(x_constrained, self.feature_names)
        
        # 3. Round integer features (e.g., byte counts, packet counts)
        # We handle this again to be sure after domain clipping
        for idx in self.integer_features:
            if idx < x_constrained.shape[1]:
                x_constrained[:, idx] = np.round(x_constrained[:, idx])
        
        # 4. Final non-negative safety check for counts and durations
        x_constrained = np.maximum(x_constrained, -2.0) # Allow some negative if scaled, but respect bounds
        
        return x_constrained
    
    def evaluate_attack(self, X_clean, epsilon=0.05, threshold=0.5):
        """
        Generate adversarial examples and evaluate attack success with advanced metrics
        """
        print(f"Generating adversarial examples (epsilon={epsilon}) for {len(X_clean)} samples...")
        
        # Generate adversarial examples
        X_adv = []
        for i, x in enumerate(X_clean):
            x_adv = self.constrained_fgsm(x, epsilon=epsilon, target_label=0)
            X_adv.append(x_adv[0])
        
        X_adv = np.array(X_adv)
        
        # Evaluate
        with torch.no_grad():
            clean_scores = self.model(torch.FloatTensor(X_clean)).numpy().flatten()
            adv_scores = self.model(torch.FloatTensor(X_adv)).numpy().flatten()
        
        # Binary predictions based on threshold
        y_clean_pred = (clean_scores > threshold).astype(int)
        y_adv_pred = (adv_scores > threshold).astype(int)
        
        # We assume X_clean were all malicious (label 1)
        y_true = np.ones(len(X_clean))
        
        # Use advanced metrics
        # Note: evaluate_network_performance expects y_true for the specific set
        # For adversarial evaluation, we want to see how detection drops
        clean_metrics = evaluate_network_performance(y_true, y_clean_pred, clean_scores)
        adv_metrics = evaluate_network_performance(y_true, y_adv_pred, adv_scores)
        
        results = {
            'attack_success_rate': 1.0 - adv_metrics['recall'], # Inverted recall on malicious samples
            'clean_detection_rate': clean_metrics['recall'],
            'adv_detection_rate': adv_metrics['recall'],
            'avg_risk_reduction': float((clean_scores - adv_scores).mean()),
            'clean_metrics': clean_metrics,
            'adv_metrics': adv_metrics,
            'X_adversarial': X_adv,
            'clean_scores': clean_scores,
            'adv_scores': adv_scores
        }
        
        return results
