"""
CICIDS-2017 dataset loader.

Loads and preprocesses the CICIDS-2017 dataset from the Canadian Institute
for Cybersecurity. Handles the multi-file CSV structure, selects relevant
feature columns, applies binary labeling (BENIGN vs all attack types), and
produces train/test splits with the same interface as the NSL-KDD loader.

The CICIDS-2017 dataset must be manually downloaded from:
https://www.unb.ca/cic/datasets/ids-2017.html

Place the CSV files in: data/cicids2017/

The dataset contains one CSV per day of network capture:
  - Monday-WorkingHours.pcap_ISCX.csv
  - Tuesday-WorkingHours.pcap_ISCX.csv
  - Wednesday-workingHours.pcap_ISCX.csv
  - Thursday-WorkingHours-*.pcap_ISCX.csv
  - Friday-WorkingHours-*.pcap_ISCX.csv

Usage:
    from src.data.cicids_loader import load_cicids2017
    X_train, X_test, y_train, y_test = load_cicids2017()
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Features selected for compatibility with the existing pipeline
SELECTED_FEATURES = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean',
    'Fwd Packet Length Mean', 'Bwd Packet Length Mean',
    'Packet Length Mean', 'Packet Length Std',
    'Average Packet Size', 'Fwd Header Length',
    'SYN Flag Count', 'ACK Flag Count',
    'Down/Up Ratio', 'Init_Win_bytes_forward',
]

LABEL_COLUMN = ' Label'  # Note: CICIDS has a leading space in column name


def load_cicids2017(
    data_dir: Optional[str] = None,
    test_size: float = 0.3,
    random_state: int = 42,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess the CICIDS-2017 dataset.

    Args:
        data_dir: Directory containing CICIDS CSV files.
            Defaults to data/cicids2017/ relative to project root.
        test_size: Fraction for test split.
        random_state: Random seed for train/test split.
        max_samples: Maximum total samples to load (for faster testing).

    Returns:
        Tuple of (X_train, X_test, y_train, y_test) with:
            - Features scaled using StandardScaler
            - Binary labels: 0 = BENIGN, 1 = Attack

    Raises:
        FileNotFoundError: If no CSV files found in data_dir.
    """
    if data_dir is None:
        try:
            from src.config import CICIDS_DIR
            data_dir = CICIDS_DIR
        except ImportError:
            data_dir = os.path.join(
                os.path.dirname(__file__), '..', '..', 'data', 'cicids2017'
            )

    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}. "
            f"Please download CICIDS-2017 from "
            f"https://www.unb.ca/cic/datasets/ids-2017.html"
        )

    logger.info(f"Found {len(csv_files)} CSV files in {data_dir}")

    # Load and concatenate all CSV files
    frames = []
    for f in sorted(csv_files):
        try:
            df = pd.read_csv(f, low_memory=False)
            frames.append(df)
            logger.info(f"  Loaded {os.path.basename(f)}: {len(df)} rows")
        except Exception as e:
            logger.warning(f"  Failed to load {os.path.basename(f)}: {e}")

    if not frames:
        raise ValueError("No valid CSV files could be loaded")

    df = pd.concat(frames, ignore_index=True)
    logger.info(f"Total rows: {len(df)}")

    # Select features that exist in the dataset
    available = [c for c in SELECTED_FEATURES if c in df.columns]
    if len(available) < 6:
        # Try with stripped column names
        df.columns = df.columns.str.strip()
        available = [c for c in [f.strip() for f in SELECTED_FEATURES]
                     if c in df.columns]

    if not available:
        raise ValueError(
            f"None of the selected features found. "
            f"Available columns: {list(df.columns[:20])}"
        )

    logger.info(f"Using {len(available)} features: {available}")

    # Binary labeling
    label_col = LABEL_COLUMN.strip() if LABEL_COLUMN.strip() in df.columns else LABEL_COLUMN
    df['binary_label'] = (df[label_col].str.strip() != 'BENIGN').astype(int)

    # Clean data
    X = df[available].copy()
    y = df['binary_label'].values

    # Replace inf and NaN
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

    # Convert to float
    X = X.values.astype(np.float32)

    # Subsample if requested
    if max_samples and len(X) > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), max_samples, replace=False)
        X = X[idx]
        y = y[idx]

    # Scale features
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    logger.info(f"Train attacks: {y_train.sum()}/{len(y_train)}")
    logger.info(f"Test attacks: {y_test.sum()}/{len(y_test)}")

    return X_train, X_test, y_train, y_test
