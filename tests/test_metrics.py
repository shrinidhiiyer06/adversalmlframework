import pytest
import numpy as np
from src.core.metrics import calculate_evasion_rate, get_perturbation_norms

def test_evasion_rate_zeros():
    y_true = np.array([1, 1, 1])
    y_pred = np.array([1, 1, 1])
    assert calculate_evasion_rate(y_true, y_pred) == 0.0

def test_evasion_rate_full():
    y_true = np.array([1, 1, 1])
    y_pred = np.array([0, 0, 0])
    assert calculate_evasion_rate(y_true, y_pred) == 1.0

def test_perturbation_norms():
    X_orig = np.array([[1.0, 1.0]])
    X_adv = np.array([[1.1, 0.9]])
    l2, linf = get_perturbation_norms(X_orig, X_adv)
    # Distance = sqrt(0.1^2 + (-0.1)^2) = sqrt(0.02) approx 0.1414
    assert np.isclose(l2, 0.1414, atol=1e-3)
    assert np.isclose(linf, 0.1, atol=1e-3)
