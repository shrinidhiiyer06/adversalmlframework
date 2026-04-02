"""
Results reporting and export module.

Provides functionality to export research findings to structured JSON files
for reproducibility and large-scale experimentation analysis.
"""

import json
import os
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


def convert_to_serializable(obj: Any) -> Any:
    """
    Convert numpy types to Python native types for JSON serialization.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def validate_results(summary: Dict, results_log: List[Dict]) -> None:
    """
    Validate results before export.
    
    Args:
        summary: Summary statistics dictionary
        results_log: List of per-run results
        
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(summary, dict):
        raise ValueError(f"summary must be dict, got {type(summary)}")
    
    if not isinstance(results_log, list):
        raise ValueError(f"results_log must be list, got {type(results_log)}")
    
    if len(summary) == 0:
        logger.warning("Empty summary dictionary")
    
    if len(results_log) == 0:
        logger.warning("Empty results log")
    
    # Check for required fields
    expected_fields = ['mean_evasion_base', 'mean_evasion_def', 'mean_robust_acc_def']
    missing = [f for f in expected_fields if f not in summary]
    if missing:
        logger.warning(f"Summary missing recommended fields: {missing}")


def export_results_to_json(
    summary: Dict[str, Any],
    results_log: List[Dict[str, Any]],
    output_dir: str = "results",
    filename: Optional[str] = None,
    version: str = "v2.0-fixed"
) -> str:
    """
    Save research findings to a structured JSON file.
    
    Critical for reproducibility and large-scale experimentation. Converts
    all numpy types to native Python types for JSON compatibility.
    
    Args:
        summary: Dictionary of aggregate metrics (mean, std, p-values, etc.)
        results_log: List of per-run result dictionaries
        output_dir: Directory to save results (default: "results")
        filename: Optional custom filename (default: auto-generated with timestamp)
        version: Version string to include in metadata
        
    Returns:
        Full filepath to the saved JSON file
        
    Raises:
        ValueError: If inputs are invalid
        IOError: If file cannot be written
        
    Example:
        >>> summary = {'mean_acc': 0.95, 'std_acc': 0.02}
        >>> results_log = [{'acc': 0.94, 'loss': 0.1}, {'acc': 0.96, 'loss': 0.09}]
        >>> filepath = export_results_to_json(summary, results_log)
        >>> print(f"Saved to {filepath}")
    """
    # Validate inputs
    validate_results(summary, results_log)
    
    # Create output directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create output directory {output_dir}: {e}")
        raise IOError(f"Cannot create output directory: {e}") from e
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"experiment_{timestamp}.json"
    
    # Ensure .json extension
    if not filename.endswith('.json'):
        filename += '.json'
    
    filepath = os.path.join(output_dir, filename)
    
    try:
        # Convert all numpy types to native Python types
        serializable_summary = convert_to_serializable(summary)
        serializable_log = convert_to_serializable(results_log)
        
        # Build report structure
        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "version": version,
                "num_runs": len(results_log),
                "output_dir": output_dir,
                "filename": filename
            },
            "summary": serializable_summary,
            "per_run_data": serializable_log
        }
        
        # Add data quality metrics
        if results_log:
            report["metadata"]["data_quality"] = {
                "complete_runs": len([r for r in results_log if all(v is not None for v in r.values())]),
                "fields_per_run": len(results_log[0]) if results_log else 0
            }
        
        # Write to file with pretty printing
        with open(filepath, "w") as f:
            json.dump(report, f, indent=4, sort_keys=True)
        
        # Verify file was written
        if not os.path.exists(filepath):
            raise IOError(f"File was not created: {filepath}")
        
        file_size = os.path.getsize(filepath)
        logger.info(f"Research report exported to {filepath} ({file_size} bytes)")
        
        return filepath
        
    except (TypeError, ValueError) as e:
        logger.error(f"Serialization failed: {e}")
        raise ValueError(f"Cannot serialize results: {e}") from e
    except IOError as e:
        logger.error(f"File write failed: {e}")
        raise IOError(f"Cannot write to {filepath}: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error during export: {e}")
        raise RuntimeError(f"Export failed: {e}") from e


def load_results_from_json(filepath: str) -> Dict[str, Any]:
    """
    Load previously exported results from JSON file.
    
    Args:
        filepath: Path to JSON results file
        
    Returns:
        Dictionary with 'metadata', 'summary', and 'per_run_data' keys
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON is invalid
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Results file not found: {filepath}")
    
    try:
        with open(filepath, 'r') as f:
            report = json.load(f)
        
        # Validate structure
        required_keys = ['metadata', 'summary', 'per_run_data']
        missing = [k for k in required_keys if k not in report]
        if missing:
            raise ValueError(f"Invalid results file, missing keys: {missing}")
        
        logger.info(f"Loaded results from {filepath}")
        return report
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {filepath}: {e}")
        raise ValueError(f"Cannot parse JSON: {e}") from e
    except Exception as e:
        logger.error(f"Error loading results: {e}")
        raise RuntimeError(f"Cannot load results: {e}") from e


def compare_experiments(
    filepath1: str,
    filepath2: str,
    metric: str = 'mean_evasion_def'
) -> Dict[str, Any]:
    """
    Compare two experiment results.
    
    Args:
        filepath1: Path to first experiment
        filepath2: Path to second experiment
        metric: Metric to compare (default: 'mean_evasion_def')
        
    Returns:
        Dictionary with comparison statistics
    """
    exp1 = load_results_from_json(filepath1)
    exp2 = load_results_from_json(filepath2)
    
    val1 = exp1['summary'].get(metric)
    val2 = exp2['summary'].get(metric)
    
    if val1 is None or val2 is None:
        raise ValueError(f"Metric '{metric}' not found in one or both experiments")
    
    diff = val2 - val1
    pct_change = (diff / val1) * 100 if val1 != 0 else float('inf')
    
    return {
        'experiment1': {
            'file': filepath1,
            'timestamp': exp1['metadata']['timestamp'],
            'value': val1
        },
        'experiment2': {
            'file': filepath2,
            'timestamp': exp2['metadata']['timestamp'],
            'value': val2
        },
        'comparison': {
            'metric': metric,
            'difference': diff,
            'percent_change': pct_change,
            'improvement': diff < 0 if 'evasion' in metric.lower() else diff > 0
        }
    }
