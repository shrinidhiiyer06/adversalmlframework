import pytest
import numpy as np
from src.evaluation.statistics import calculate_statistical_significance, calculate_confidence_interval

def test_p_value_significance():
    # Large difference
    baseline = [0.9, 0.9, 0.9]
    defended = [0.1, 0.1, 0.15]
    res = calculate_statistical_significance(baseline, defended)
    assert res['p_value'] < 0.05
    assert res['is_significant'] is True

def test_ci_bounds():
    data = [0.1, 0.12, 0.11, 0.09, 0.1]
    margin = calculate_confidence_interval(data)
    assert margin > 0
    assert margin < 0.1 # reasonable margin
