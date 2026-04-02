"""
Train Baseline Network Risk Classifier
Trains intrusion detection model on NSL-KDD dataset
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.network_loader import NetworkDataLoader
from src.risk_engine.network_classifier import NetworkRiskClassifier


def train_network_classifier():
    """Train network intrusion detection classifier"""
    
    print("="*60)
    print("Training Network Risk Classifier on NSL-KDD Dataset")
    print("="*60)
    
    # Load data
    loader = NetworkDataLoader()
    
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    train_path = os.path.join(data_dir, 'KDDTrain+.txt')
    test_path = os.path.join(data_dir, 'KDDTest+.txt')
    
    print("\n[1/5] Loading training data...")
    X_train, y_train, _ = loader.load_and_preprocess(train_path, is_train=True)
    
    print("\n[2/5] Loading test data...")
    X_test, y_test, _ = loader.load_and_preprocess(test_path, is_train=False)
    
    # Save preprocessors
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    loader.save_preprocessors(models_dir)
    
    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1)
    
    # Create dataloaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    print(f"\n[3/5] Initializing model...")
    print(f"Input features: {X_train.shape[1]}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Initialize model
    model = NetworkRiskClassifier(input_dim=X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print(f"\n[4/5] Training for 20 epochs...")
    print("-"*60)
    
    epochs = 20
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_t)
            test_preds = (test_outputs > 0.5).float()
            accuracy = (test_preds == y_test_t).float().mean()
            
            # Calculate precision and recall for attack class
            true_positives = ((test_preds == 1) & (y_test_t == 1)).sum().float()
            false_positives = ((test_preds == 1) & (y_test_t == 0)).sum().float()
            false_negatives = ((test_preds == 0) & (y_test_t == 1)).sum().float()
            
            precision = true_positives / (true_positives + false_positives + 1e-10)
            recall = true_positives / (true_positives + false_negatives + 1e-10)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        avg_loss = total_loss / len(train_loader)
        
        print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | "
              f"Acc: {accuracy:.4f} | Prec: {precision:.4f} | "
              f"Rec: {recall:.4f} | F1: {f1:.4f}")
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model_path = os.path.join(models_dir, 'network_risk_classifier.pth')
            torch.save(model.state_dict(), model_path)
    
    print("-"*60)
    print(f"\n[5/5] Training complete!")
    print(f"Best test accuracy: {best_accuracy:.4f}")
    print(f"Model saved to: {model_path}")
    print("="*60)
    
    return model


if __name__ == "__main__":
    train_network_classifier()
