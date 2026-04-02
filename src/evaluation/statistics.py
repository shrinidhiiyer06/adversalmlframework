"""
Statistical analysis module for research rigor.

Provides statistical tests and confidence intervals for comparing
baseline and defended model performance. All functions handle edge
cases and include numerical stability improvements.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

# Numerical stability epsilon
EPSILON = 1e-10


def validate_statistical_data(data: List[float], name: str = "data") -> None:
    """
    Validate data for statistical analysis.
    
    Args:
        data: List of numeric values
        name: Name for error messages
        
    Raises:
        ValueError: If validation fails
    """
    if not data:
        raise ValueError(f"{name} cannot be empty")
    
    if len(data) < 2:
        raise ValueError(f"{name} must have at least 2 samples for statistical analysis, got {len(data)}")
    
    if not all(isinstance(x, (int, float, np.number)) for x in data):
        raise TypeError(f"{name} must contain only numeric values")
    
    if not all(np.isfinite(x) for x in data):
        raise ValueError(f"{name} contains non-finite values (NaN or inf)")


def calculate_confidence_interval(
    data: List[float],
    confidence: float = 0.95
) -> float:
    """
    Compute confidence interval margin for a metric.
    
    Uses t-distribution for small samples (n < 30) and normal distribution
    for large samples. Returns the margin of error (half-width of CI).
    
    Args:
        data: List of metric values (e.g., accuracies from multiple runs)
        confidence: Confidence level (default: 0.95 for 95% CI)
        
    Returns:
        Margin of error. The confidence interval is:
            [mean - margin, mean + margin]
            
    Example:
        >>> data = [0.85, 0.87, 0.86, 0.88, 0.84]
        >>> margin = calculate_confidence_interval(data)
        >>> mean = np.mean(data)
        >>> print(f"Mean: {mean:.3f}, 95% CI: [{mean-margin:.3f}, {mean+margin:.3f}]")
        Mean: 0.860, 95% CI: [0.845, 0.875]
        
    Notes:
        - For n < 2, returns 0.0 (cannot compute CI)
        - Uses t-distribution for accurate small-sample inference
        - Returns 0.0 if std is zero (all values identical)
    """
    # Validate
    if len(data) < 2:
        logger.warning(f"Cannot compute CI with {len(data)} samples, returning 0.0")
        return 0.0
    
    validate_statistical_data(data, "data")
    
    if not 0 < confidence < 1:
        raise ValueError(f"Confidence must be in (0, 1), got {confidence}")
    
    # Convert to numpy array
    data_arr = np.array(data, dtype=float)
    
    n = len(data_arr)
    mean = np.mean(data_arr)
    std = np.std(data_arr, ddof=1)  # Sample standard deviation (unbiased)
    
    # If std is zero (all values identical), CI margin is zero
    if std < EPSILON:
        logger.debug("Standard deviation is ~0, returning CI margin of 0.0")
        return 0.0
    
    # Standard error of the mean
    std_err = std / np.sqrt(n)
    
    # Use t-distribution for small samples, normal for large samples
    if n < 30:
        # t-distribution with (n-1) degrees of freedom
        t_critical = stats.t.ppf((1 + confidence) / 2, df=n - 1)
        margin = t_critical * std_err
    else:
        # Normal distribution (z-score)
        z_critical = stats.norm.ppf((1 + confidence) / 2)
        margin = z_critical * std_err
    
    return float(margin)


def calculate_statistical_significance(
    baseline_scores: List[float],
    defended_scores: List[float],
    alpha: float = 0.01
) -> Dict[str, float]:
    """
    Perform paired t-test and compute effect size (Cohen's d).
    
    Tests whether the difference between baseline and defended scores
    is statistically significant. Uses paired t-test since the same
    samples are evaluated in both conditions.
    
    Args:
        baseline_scores: Metric values from baseline model
        defended_scores: Metric values from defended model
        alpha: Significance level (default: 0.01 for 99% confidence)
        
    Returns:
        Dictionary containing:
            - t_statistic: T-test statistic
            - p_value: Probability of observing this difference by chance
            - cohens_d: Effect size (standardized mean difference)
            - is_significant: Whether p_value < alpha
            - interpretation: Human-readable interpretation of effect size
            
    Example:
        >>> baseline = [0.9, 0.88, 0.91]
        >>> defended = [0.1, 0.12, 0.09]
        >>> result = calculate_statistical_significance(baseline, defended)
        >>> print(f"p-value: {result['p_value']:.4f}")
        >>> print(f"Significant: {result['is_significant']}")
        >>> print(f"Effect size: {result['interpretation']}")
        
    Notes:
        - Paired t-test is appropriate when comparing same samples under different conditions
        - Cohen's d interpretation: 0.2=small, 0.5=medium, 0.8=large effect
        - Negative Cohen's d means defended performs better (for evasion rate)
    """
    # Validate inputs
    validate_statistical_data(baseline_scores, "baseline_scores")
    validate_statistical_data(defended_scores, "defended_scores")
    
    if len(baseline_scores) != len(defended_scores):
        raise ValueError(
            f"Baseline and defended scores must have same length: "
            f"{len(baseline_scores)} != {len(defended_scores)}"
        )
    
    if not 0 < alpha < 1:
        raise ValueError(f"Alpha must be in (0, 1), got {alpha}")
    
    # Convert to numpy arrays
    baseline = np.array(baseline_scores, dtype=float)
    defended = np.array(defended_scores, dtype=float)
    
    try:
        # Perform paired t-test
        t_stat, p_value = stats.ttest_rel(baseline, defended)
        
        # Handle NaN results (e.g., if all values are identical)
        if not np.isfinite(t_stat):
            logger.warning("T-test returned non-finite result, setting t_stat=0")
            t_stat = 0.0
        
        if not np.isfinite(p_value):
            logger.warning("T-test returned non-finite p-value, setting p_value=1.0")
            p_value = 1.0
        
        # Compute Cohen's d (effect size)
        # Cohen's d = (mean difference) / (std of differences)
        diff = defended - baseline
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        
        if std_diff < EPSILON:
            # All differences are zero or nearly zero
            cohens_d = 0.0
            logger.debug("Standard deviation of differences is ~0, setting Cohen's d = 0")
        else:
            cohens_d = mean_diff / std_diff
        
        # Interpret effect size
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            interpretation = "negligible"
        elif abs_d < 0.5:
            interpretation = "small"
        elif abs_d < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"
        
        result = {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": float(cohens_d),
            "is_significant": bool(p_value < alpha),
            "alpha": alpha,
            "interpretation": interpretation,
            "n_samples": len(baseline),
            "mean_baseline": float(np.mean(baseline)),
            "mean_defended": float(np.mean(defended)),
            "mean_difference": float(mean_diff),
            "degrees_of_freedom": len(baseline) - 1,
            "null_hypothesis": "Zero-Trust policies do not improve adversarial robustness over ML-only baseline"
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Statistical significance calculation failed: {e}")
        raise RuntimeError(f"Cannot compute statistical significance: {e}") from e


def calculate_multiple_comparison_correction(
    p_values: List[float],
    method: str = 'bonferroni'
) -> List[float]:
    """
    Apply multiple comparison correction to p-values.
    
    When running multiple statistical tests, the chance of false positives
    increases. This function applies corrections to maintain overall
    significance level.
    
    Args:
        p_values: List of p-values from multiple tests
        method: Correction method - 'bonferroni' or 'holm'
        
    Returns:
        List of corrected p-values
        
    Example:
        >>> p_values = [0.01, 0.03, 0.05, 0.02]
        >>> corrected = calculate_multiple_comparison_correction(p_values)
        >>> print(corrected)
        [0.04, 0.12, 0.20, 0.08]  # Bonferroni correction
    """
    if not p_values:
        return []
    
    n = len(p_values)
    p_array = np.array(p_values)
    
    if method == 'bonferroni':
        # Bonferroni: multiply by number of tests
        corrected = p_array * n
        corrected = np.minimum(corrected, 1.0)  # Cap at 1.0
        
    elif method == 'holm':
        # Holm-Bonferroni: step-down procedure
        indices = np.argsort(p_array)
        corrected = np.zeros_like(p_array)
        
        for i, idx in enumerate(indices):
            corrected[idx] = min(p_array[idx] * (n - i), 1.0)
            if i > 0:
                corrected[idx] = max(corrected[idx], corrected[indices[i-1]])
    else:
        raise ValueError(f"Unknown method: {method}. Use 'bonferroni' or 'holm'")
    
    return corrected.tolist()


def calculate_power_analysis(
    effect_size: float,
    n_samples: int,
    alpha: float = 0.05,
    alternative: str = 'two-sided'
) -> float:
    """
    Calculate statistical power for a t-test.
    
    Power is the probability of detecting an effect when it exists.
    Higher power = lower chance of false negatives.
    
    Args:
        effect_size: Expected Cohen's d
        n_samples: Number of samples per group
        alpha: Significance level
        alternative: 'two-sided', 'larger', or 'smaller'
        
    Returns:
        Statistical power (probability of detecting the effect)
        
    Example:
        >>> power = calculate_power_analysis(effect_size=0.5, n_samples=30)
        >>> print(f"Power: {power:.2%}")
        Power: 57.40%
    """
    from scipy.stats import nct
    
    if n_samples < 2:
        raise ValueError("n_samples must be >= 2")
    
    df = 2 * n_samples - 2  # Degrees of freedom for two-sample t-test
    
    # Non-centrality parameter
    ncp = effect_size * np.sqrt(n_samples / 2)
    
    # Critical value
    if alternative == 'two-sided':
        t_crit = stats.t.ppf(1 - alpha/2, df)
    elif alternative == 'larger':
        t_crit = stats.t.ppf(1 - alpha, df)
    else:  # smaller
        t_crit = stats.t.ppf(alpha, df)
    
    # Power calculation using non-central t-distribution
    if alternative == 'two-sided':
        power = 1 - nct.cdf(t_crit, df, ncp) + nct.cdf(-t_crit, df, ncp)
    elif alternative == 'larger':
        power = 1 - nct.cdf(t_crit, df, ncp)
    else:
        power = nct.cdf(t_crit, df, -ncp)
    
    return float(power)


def generate_statistical_report(
    baseline_scores: List[float],
    defended_scores: List[float],
    metric_name: str = "evasion_rate"
) -> pd.DataFrame:
    """
    Generate a comprehensive statistical comparison report.
    
    Args:
        baseline_scores: Scores from baseline model
        defended_scores: Scores from defended model  
        metric_name: Name of the metric being compared
        
    Returns:
        DataFrame with statistical analysis results
    """
    validate_statistical_data(baseline_scores, "baseline_scores")
    validate_statistical_data(defended_scores, "defended_scores")
    
    # Basic statistics
    baseline_mean = np.mean(baseline_scores)
    baseline_std = np.std(baseline_scores, ddof=1)
    defended_mean = np.mean(defended_scores)
    defended_std = np.std(defended_scores, ddof=1)
    
    # Statistical test
    sig_result = calculate_statistical_significance(baseline_scores, defended_scores)
    
    # Confidence intervals
    baseline_ci = calculate_confidence_interval(baseline_scores)
    defended_ci = calculate_confidence_interval(defended_scores)
    
    # Create report
    report = pd.DataFrame({
        'Metric': [metric_name],
        'Baseline_Mean': [baseline_mean],
        'Baseline_Std': [baseline_std],
        'Baseline_CI_95': [f"±{baseline_ci:.4f}"],
        'Defended_Mean': [defended_mean],
        'Defended_Std': [defended_std],
        'Defended_CI_95': [f"±{defended_ci:.4f}"],
        'Improvement': [defended_mean - baseline_mean],
        'P_Value': [sig_result['p_value']],
        'Cohens_D': [sig_result['cohens_d']],
        'Effect_Size': [sig_result['interpretation']],
        'Significant': [sig_result['is_significant']]
    })
    
    return report


def format_ci(mean: float, std: float, n: int, confidence: float = 0.95) -> str:
    """Format a standard mean +/- std and 95% CI string."""
    from scipy import stats
    import numpy as np
    ci_margin = stats.t.ppf((1 + confidence) / 2, df=n - 1) * (std / np.sqrt(n))
    return f"{mean:.4f} ± {std:.4f} (95% CI: [{mean - ci_margin:.4f}, {mean + ci_margin:.4f}])"


def multi_seed_aggregate(all_metrics: List[Dict[str, float]]) -> Dict[str, Dict]:
    """Compute mean, std, and 95% CI across seeds."""
    from scipy import stats
    import numpy as np
    metric_names = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    n = len(all_metrics)
    agg = {}

    for name in metric_names:
        values = [m[name] for m in all_metrics if name in m]
        if not values:
            continue
        mean = np.mean(values)
        std = np.std(values, ddof=1)  # Sample std
        ci_margin = stats.t.ppf((1 + 0.95) / 2, df=n - 1) * std / np.sqrt(n)

        agg[name] = {
            'mean': float(mean),
            'std': float(std),
            'ci_95_lower': float(mean - ci_margin),
            'ci_95_upper': float(mean + ci_margin),
            'ci_95_margin': float(ci_margin),
            'per_seed': values,
            'formatted': f"{mean:.4f} ± {std:.4f} (95% CI: [{mean - ci_margin:.4f}, {mean + ci_margin:.4f}])"
        }

    return agg
