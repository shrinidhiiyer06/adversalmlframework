import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import os

def train_models():
    print("Loading dataset...")
    # Construct absolute paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "..", "data")
    model_dir = os.path.join(base_dir, "..", "..", "models")
    
    data_path = os.path.join(data_dir, "combined_traffic.csv")
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return

    df = pd.read_csv(data_path)
        
    X = df.drop(columns=['label'])
    y = df['label']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print("-" * 50)
    print("Training Model 1: Isolation Forest (Unsupervised / Anomaly Detection)")
    # Isolation Forest assumes anomalies are rare and different.
    # CRITICAL UPGRADE: Train ONLY on Benign data (y_train == 0)
    # This teaches the model what "Normal" looks like, so it rejects everything else.
    X_train_benign = X_train[y_train == 0]
    print(f"Training Isolation Forest on {len(X_train_benign)} benign samples out of {len(X_train)} total training samples.")
    
    iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42) # Lower contamination since we train on clean data
    iso_forest.fit(X_train_benign)
    
    # Evaluation
    # IF returns -1 for anomaly, 1 for normal
    y_pred_iso = iso_forest.predict(X_test)
    # Map: -1 -> 1 (Attack), 1 -> 0 (Benign)
    y_pred_iso_mapped = [1 if x == -1 else 0 for x in y_pred_iso]
    
    print("Isolation Forest Results (Anomaly Detection):")
    print(classification_report(y_test, y_pred_iso_mapped))
    
    print("-" * 50)
    print("Training Model 2: Random Forest (Supervised Classification)")
    # RF is trained on BOTH benign and attack data to learn specific attack signatures
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred_rf = rf.predict(X_test)
    print("Random Forest Results (Supervised):")
    print(classification_report(y_test, y_pred_rf))
    print(f"ROC-AUC: {roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]):.4f}")
    
    # --- COMPARISON METRICS (Upgrade 3) ---
    print("-" * 50)
    print("Model Performance Comparison:")
    # Calculate False Positive Rate for IF (Benign classified as Attack)
    benign_mask = (y_test == 0)
    if_fp = np.sum(np.array(y_pred_iso_mapped)[benign_mask] == 1)
    if_fp_rate = if_fp / np.sum(benign_mask) if np.sum(benign_mask) > 0 else 0
    print(f"Isolation Forest False Positive Rate: {if_fp_rate:.4f}")

    # Calculate Detection Rate for IF on Attacks
    attack_mask = (y_test == 1)
    if_dr = np.sum(np.array(y_pred_iso_mapped)[attack_mask] == 1)
    if_dr_rate = if_dr / np.sum(attack_mask) if np.sum(attack_mask) > 0 else 0
    print(f"Isolation Forest Attack Detection Rate: {if_dr_rate:.4f}")
    print("-" * 50)
    
    # Save models
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(iso_forest, os.path.join(model_dir, "isolation_forest.pkl"))
    joblib.dump(rf, os.path.join(model_dir, "random_forest.pkl"))
    
    # Save test data for explainability and evaluation
    test_set = X_test.copy()
    test_set['label'] = y_test
    test_set.to_csv(os.path.join(model_dir, "test_set.csv"), index=False)
    
    # Save training data for surrogate training (Research Rigor: Prevent Leakage)
    train_set = X_train.copy()
    train_set['label'] = y_train
    train_set.to_csv(os.path.join(model_dir, "train_set.csv"), index=False)
    
    # Save feature bounds for ART clipping (Research Rigor: Realistic Constraints)
    # Store as a list of (min, max) for each feature
    feature_min = X_train.min().values.astype(np.float32)
    feature_max = X_train.max().values.astype(np.float32)
    joblib.dump((feature_min, feature_max), os.path.join(model_dir, "feature_bounds.pkl"))

    print(f"Models and data saved to {model_dir}")

if __name__ == "__main__":
    train_models()
