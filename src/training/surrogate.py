import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from copy import deepcopy
import logging

from src.config import SURROGATE_LR, SURROGATE_EPOCHS, EARLY_STOPPING_PATIENCE, TRAIN_VAL_SPLIT

logger = logging.getLogger(__name__)

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.model(x)

def train_surrogate(X, y, epochs=SURROGATE_EPOCHS, seed=42):
    """
    Trains a surrogate neural network with research-grade rigor:
    - Stratified Train/Val split
    - Early Stopping based on validation accuracy
    - Model Checkpointing (saves best weights)
    """
    logger.info("Initializing surrogate training with early stopping...")
    
    # Stratified Split (ensures class balance in validation)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TRAIN_VAL_SPLIT, shuffle=True, stratify=y, random_state=seed
    )
    
    model = SimpleNN(X.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=SURROGATE_LR)
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        
        # Validation Phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_preds = torch.argmax(val_outputs, dim=1).numpy()
            val_acc = accuracy_score(y_val, val_preds)
            
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = deepcopy(model.state_dict())
            patience_counter = 0
            if epoch % 10 == 0:
                logger.debug(f"Epoch {epoch}: New best Val Acc: {val_acc*100:.2f}%")
        else:
            patience_counter += 1
            
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            logger.info(f"Early stopping triggered at epoch {epoch}. Best Val Acc: {best_val_acc*100:.2f}%")
            break
            
    # Load best weights
    if best_model_state:
        model.load_state_dict(best_model_state)
        
    return model, best_val_acc
