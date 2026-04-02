"""
Network Risk Classifier
ML-based intrusion detection model for network traffic
"""

import torch
import torch.nn as nn


class NetworkRiskClassifier(nn.Module):
    """
    Neural network for network intrusion detection
    Outputs risk score (0-1) for network flows
    """
    
    def __init__(self, input_dim=41):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """Forward pass through network"""
        return self.network(x)
    
    def get_risk_score(self, x):
        """
        Returns risk score between 0-1 for a network flow
        
        Args:
            x: Input tensor of network flow features
            
        Returns:
            Risk score (0=benign, 1=malicious)
        """
        with torch.no_grad():
            score = self.forward(x)
            return score.item() if score.dim() == 0 else score.squeeze()
    
    def predict_batch(self, X):
        """
        Predict risk scores for a batch of network flows
        
        Args:
            X: Batch of network flow features (numpy array or tensor)
            
        Returns:
            Risk scores as numpy array
        """
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        
        with torch.no_grad():
            scores = self.forward(X)
            return scores.squeeze().numpy()
