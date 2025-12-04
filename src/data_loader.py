"""
Data loading and preprocessing module for NIFTY intraday data.
"""

import pandas as pd
import numpy as np
import os


def load_data(filepath):
    """
    Load CSV from data/raw/ folder and perform basic preprocessing.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe with parsed timestamps
    """
    print(f"Loading data from {filepath}...")
    
    # Read CSV
    df = pd.read_csv(filepath)
    
    # Check column names and rename if needed
    column_mapping = {
        'timestamp': 'Timestamp',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }
    
    # Rename columns if they exist in lowercase
    df.columns = df.columns.str.strip()
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df = df.rename(columns={old_col: new_col})
    
    # Parse Timestamp column as datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Sort by Timestamp in ascending order
    df = df.sort_values('Timestamp').reset_index(drop=True)
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.any():
        print("Warning: Missing values detected:")
        print(missing_values[missing_values > 0])
        df = df.dropna()
        print(f"Dropped {missing_values.sum()} rows with missing values")
    
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df


def create_target_column(df):
    """
    Generate target variable for prediction.
    Target = 1 if next_close > current_close, else 0
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLC data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with 'target' column added
    """
    print("Creating target column...")
    
    # Create next_close column
    df['next_close'] = df['Close'].shift(-1)
    
    # Create target: 1 if next close is higher, 0 otherwise
    df['target'] = (df['next_close'] > df['Close']).astype(int)
    
    # Drop the last row (no target available)
    df = df[:-1].copy()
    
    # Drop helper column
    df = df.drop('next_close', axis=1)
    
    # Check target distribution
    target_dist = df['target'].value_counts()
    print(f"Target distribution:\n{target_dist}")
    print(f"Class balance: {target_dist[1]/len(df)*100:.2f}% positive class")
    
    return df


def train_test_split_timeseries(df, train_ratio=0.7):
    """
    Split data chronologically (no shuffle) for time series.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features and target
    train_ratio : float
        Proportion of data for training (default 0.7)
        
    Returns:
    --------
    tuple
        X_train, X_test, y_train, y_test, test_df (with Timestamp)
    """
    print(f"\nSplitting data with train_ratio={train_ratio}...")
    
    # Calculate split index
    split_index = int(len(df) * train_ratio)
    
    # Split train and test
    train = df[:split_index].copy()
    test = df[split_index:].copy()
    
    print(f"Train size: {len(train)} samples")
    print(f"Test size: {len(test)} samples")
    
    # Separate features and target
    # Drop non-feature columns: target, Timestamp, symbol, id, exchange
    non_feature_cols = ['target', 'Timestamp', 'symbol', 'id', 'exchange']
    feature_cols = [col for col in df.columns if col not in non_feature_cols]
    
    X_train = train[feature_cols]
    y_train = train['target']
    
    X_test = test[feature_cols]
    y_test = test['target']
    
    # Keep test dataframe with Timestamp for PnL calculation
    test_df = test.copy()
    
    print(f"Features shape: X_train={X_train.shape}, X_test={X_test.shape}")
    print(f"Target shape: y_train={y_train.shape}, y_test={y_test.shape}")
    
    return X_train, X_test, y_train, y_test, test_df


def save_processed_data(train_df, test_df, output_dir='data/processed'):
    """
    Save train and test datasets for reproducibility.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training dataframe
    test_df : pd.DataFrame
        Test dataframe
    output_dir : str
        Directory to save processed data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'train.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\nProcessed data saved:")
    print(f"  Train: {train_path}")
    print(f"  Test: {test_path}")
