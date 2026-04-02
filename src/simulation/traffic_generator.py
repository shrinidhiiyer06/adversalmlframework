import pandas as pd
import numpy as np
import time
import random
import os
from datetime import datetime

def generate_benign_traffic(num_samples=1000):
    """
    Simulates benign traffic for a Zero-Trust Network.
    Returns a DataFrame with network and behavioral features.
    """
    print(f"Generating {num_samples} benign traffic samples...")
    
    data = []
    
    for _ in range(num_samples):
        # 1. Network Features
        packet_size = int(np.random.normal(loc=500, scale=150)) # Bytes, normal dist
        packet_size = max(64, packet_size) # Min Ethernet frame size
        
        flow_duration = np.random.exponential(scale=2.0) # Seconds
        
        # 2. Behavioral Features
        # Time between requests from same user (simulated)
        request_frequency = np.random.poisson(lam=5) # Requests per minute
        
        # 3. Authentication/Context Features
        # Entropy of the auth token (Benign usually has high, consistent entropy)
        token_entropy = np.random.normal(loc=7.5, scale=0.2) 
        token_entropy = min(8.0, max(0.0, token_entropy))
        
        # Geo-velocity: speed of travel between logins (Benign = low/realistic)
        geo_velocity = np.random.exponential(scale=10) # km/h (mostly stationary/slow)
        
        # Role Trust Score: Policy engine's existing trust in user (0-100)
        trust_score = int(np.random.normal(loc=80, scale=10))
        trust_score = min(100, max(0, trust_score))
        
        # 4. Label
        label = 0 # 0 = Benign
        
        data.append({
            "packet_size": packet_size,
            "flow_duration": flow_duration,
            "request_frequency": request_frequency,
            "token_entropy": token_entropy,
            "geo_velocity": geo_velocity,
            "trust_score": trust_score,
            "label": label
        })
        
    df = pd.DataFrame(data)
    print("Benign traffic generation complete.")
    return df

if __name__ == "__main__":
    df = generate_benign_traffic(2000)
    
    # Construct absolute path to data directory
    # src/simulation/traffic_generator.py -> src/simulation -> src -> root -> data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "..", "..", "data")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "benign_traffic.csv")
    
    df.to_csv(output_path, index=False)
    print(f"Saved benign traffic to {output_path}")
