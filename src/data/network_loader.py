"""
Network Data Loader for NSL-KDD Dataset
Handles loading and preprocessing of network intrusion detection data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os


class NetworkDataLoader:
    """Loads and preprocesses NSL-KDD network traffic dataset"""
    
    def __init__(self):
        self.feature_names = [
            'duration', 'protocol_type', 'service', 'flag',
            'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
            'urgent', 'hot', 'num_failed_logins', 'logged_in',
            'num_compromised', 'root_shell', 'su_attempted',
            'num_root', 'num_file_creations', 'num_shells',
            'num_access_files', 'num_outbound_cmds',
            'is_host_login', 'is_guest_login', 'count',
            'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
            'diff_srv_rate', 'srv_diff_host_rate',
            'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
            'label', 'difficulty'
        ]
        
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.categorical_cols = ['protocol_type', 'service', 'flag']
        
    def load_and_preprocess(self, filepath, is_train=True):
        """
        Load and preprocess NSL-KDD dataset
        
        Args:
            filepath: Path to KDDTrain+.txt or KDDTest+.txt
            is_train: Whether this is training data (fit encoders) or test data (transform only)
            
        Returns:
            X_scaled: Preprocessed feature array
            y_binary: Binary labels (0=normal, 1=attack)
            y_original: Original attack type labels
        """
        print(f"Loading data from {filepath}...")
        
        # Load data
        df = pd.read_csv(filepath, names=self.feature_names, header=None)
        
        print(f"Loaded {len(df)} samples")
        
        # Separate features and labels
        X = df.drop(['label', 'difficulty'], axis=1)
        y = df['label']
        
        # Binary classification: normal vs attack
        y_binary = (y != 'normal').astype(int)
        
        print(f"Normal samples: {(y_binary == 0).sum()}, Attack samples: {(y_binary == 1).sum()}")
        
        # Encode categorical features
        for col in self.categorical_cols:
            if is_train:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col])
            else:
                # Handle unseen categories in test set
                X[col] = X[col].apply(
                    lambda x: x if x in self.label_encoders[col].classes_ 
                    else self.label_encoders[col].classes_[0]
                )
                X[col] = self.label_encoders[col].transform(X[col])
        
        # Convert to numpy array
        X_array = X.values.astype(np.float32)
        
        # Scale numerical features
        if is_train:
            X_scaled = self.scaler.fit_transform(X_array)
        else:
            X_scaled = self.scaler.transform(X_array)
        
        print(f"Preprocessed data shape: {X_scaled.shape}")
        
        return X_scaled, y_binary.values, y.values
    
    def save_preprocessors(self, save_dir='models'):
        """Save label encoders and scaler for later use"""
        os.makedirs(save_dir, exist_ok=True)
        
        with open(os.path.join(save_dir, 'label_encoders.pkl'), 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"Saved preprocessors to {save_dir}/")
    
    def load_preprocessors(self, save_dir='models'):
        """Load saved label encoders and scaler"""
        with open(os.path.join(save_dir, 'label_encoders.pkl'), 'rb') as f:
            self.label_encoders = pickle.load(f)
        
        with open(os.path.join(save_dir, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
        
        print(f"Loaded preprocessors from {save_dir}/")
    
    def get_feature_bounds(self, X):
        """Get min/max bounds for each feature (useful for adversarial attacks)"""
        return {
            'min': X.min(axis=0),
            'max': X.max(axis=0)
        }
