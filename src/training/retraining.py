import pandas as pd
import numpy as np
import joblib
import os
import torch
import torch.nn as nn
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.training.surrogate import SimpleNN
from src.config import MODEL_DIR, FGM_EPS

def fortify_model():
    """Retrains the Random Forest on a mix of clean and adversarial data."""
    # Load original training data
    train_df = pd.read_csv(os.path.join(MODEL_DIR, "train_set.csv"))
    X_train = train_df.drop(columns=['label']).values.astype(np.float32)
    y_train = train_df['label'].values
    clip_values = joblib.load(os.path.join(MODEL_DIR, "feature_bounds.pkl"))

    print("--- Phase 1: Training Surrogate for Data Augmentation ---")
    surrogate = SimpleNN(X_train.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(surrogate.parameters(), lr=0.01)
    
    X_tensor = torch.tensor(X_train[:2000], dtype=torch.float32)
    y_tensor = torch.tensor(y_train[:2000], dtype=torch.long)
    
    for epoch in range(30):
        optimizer.zero_grad()
        outputs = surrogate(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

    print("--- Phase 2: Generating Adversarial Training Samples (FGM) ---")
    classifier = PyTorchClassifier(
        model=surrogate, loss=criterion, optimizer=optimizer,
        input_shape=(X_train.shape[1],), nb_classes=2, clip_values=clip_values
    )
    
    # Generate adversarial examples for 20% of the training set
    aug_size = int(len(X_train) * 0.2)
    attack = FastGradientMethod(estimator=classifier, eps=FGM_EPS)
    X_adv = attack.generate(x=X_train[:aug_size])
    y_adv = y_train[:aug_size] 

    print("--- Phase 3: Merging & Retraining Fortified Random Forest ---")
    X_fortified = np.vstack([X_train, X_adv])
    y_fortified = np.hstack([y_train, y_adv])
    
    fortified_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    fortified_rf.fit(X_fortified, y_fortified)
    
    # Save the new model
    joblib.dump(fortified_rf, os.path.join(MODEL_DIR, "fortified_random_forest.pkl"))
    print(f"Fortified Model Saved. Augmented Dataset Size: {len(X_fortified)}")
    
    return True

if __name__ == "__main__":
    fortify_model()
