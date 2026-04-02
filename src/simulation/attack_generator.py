import pandas as pd
import numpy as np
import os

def generate_adversarial_traffic(benign_df, attack_type="evasion", intrusion_ratio=0.3):
    """
    Takes benign traffic and generates specific adversarial samples.
    intrusion_ratio: Percentage of data that should be malicious.
    """
    print(f"Generating '{attack_type}' attacks...")
    
    n_attacks = int(len(benign_df) * intrusion_ratio)
    attack_indices = np.random.choice(benign_df.index, n_attacks, replace=False)
    
    # Create a copy to act as our base for attacks
    attack_data = benign_df.loc[attack_indices].copy()
    
    if attack_type == "evasion":
        # Evasion: Modify traffic slightly to slip under radar
        # Improve trust score artificially to look "safer"
        attack_data['trust_score'] = attack_data['trust_score'] + np.random.normal(loc=15, scale=5)
        attack_data['trust_score'] = attack_data['trust_score'].clip(upper=100)
        
        # Reduce flow duration to look like "quick, efficient" access
        attack_data['flow_duration'] = attack_data['flow_duration'] * 0.5
        
    elif attack_type == "mimicry":
        # Mimicry: Try to look EXACTLY like a high-entropy boring request
        # But maybe the request frequency is suspicious (Brute force disguised)
        attack_data['request_frequency'] = attack_data['request_frequency'] + np.random.poisson(lam=20, size=len(attack_data))
        
        # Perfect entropy mimicry
        attack_data['token_entropy'] = np.random.normal(loc=7.5, scale=0.05, size=len(attack_data))
        
    elif attack_type == "manipulation":
        # Feature Poisoning / Manipulation
        # Strange packet sizes but valid auth
        attack_data['packet_size'] = attack_data['packet_size'] + np.random.normal(loc=1000, scale=200, size=len(attack_data))
        
    # Set Label to 1 (Adversarial)
    attack_data['label'] = 1
    
    return attack_data

if __name__ == "__main__":
    # Construct absolute path to data directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    benign_path = os.path.join(data_dir, "benign_traffic.csv")
    
    # Load benign data
    if not os.path.exists(benign_path):
        print(f"Error: Benign data not found at {benign_path}. Run traffic_generator.py first.")
        exit(1)
        
    benign_df = pd.read_csv(benign_path)
        
    # Generate different attack vectors
    evasion_attacks = generate_adversarial_traffic(benign_df, "evasion", 0.1)
    mimicry_attacks = generate_adversarial_traffic(benign_df, "mimicry", 0.1)
    manipulation_attacks = generate_adversarial_traffic(benign_df, "manipulation", 0.1)
    
    # Combine everything
    combined_df = pd.concat([benign_df, evasion_attacks, mimicry_attacks, manipulation_attacks], ignore_index=True)
    
    # Shuffle
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)
    
    output_path = os.path.join(data_dir, "combined_traffic.csv")
    combined_df.to_csv(output_path, index=False)
    print(f"Saved combined dataset to {output_path} with shape {combined_df.shape}")
