"""
Feature engineering module for creating technical indicators and features.
"""

import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange


def add_technical_indicators(df):
    """
    Calculate technical indicators using ta library.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLC data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with technical indicators added
    """
    print("Adding technical indicators...")
    
    df = df.copy()
    
    # Returns
    df['returns'] = df['Close'].pct_change()
    
    # RSI (Relative Strength Index)
    rsi_indicator = RSIIndicator(close=df['Close'], window=14)
    df['RSI_14'] = rsi_indicator.rsi()
    
    # MACD (Moving Average Convergence Divergence)
    macd_indicator = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd_indicator.macd()
    df['MACD_signal'] = macd_indicator.macd_signal()
    df['MACD_hist'] = macd_indicator.macd_diff()
    
    # Bollinger Bands
    bb_indicator = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_upper'] = bb_indicator.bollinger_hband()
    df['BB_middle'] = bb_indicator.bollinger_mavg()
    df['BB_lower'] = bb_indicator.bollinger_lband()
    df['BB_width'] = df['BB_upper'] - df['BB_lower']
    
    # Simple Moving Averages
    df['SMA_5'] = SMAIndicator(close=df['Close'], window=5).sma_indicator()
    df['SMA_10'] = SMAIndicator(close=df['Close'], window=10).sma_indicator()
    df['SMA_20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
    
    # Exponential Moving Averages
    df['EMA_5'] = EMAIndicator(close=df['Close'], window=5).ema_indicator()
    df['EMA_10'] = EMAIndicator(close=df['Close'], window=10).ema_indicator()
    df['EMA_20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
    
    # ATR (Average True Range) - volatility indicator
    atr_indicator = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    df['ATR_14'] = atr_indicator.average_true_range()
    
    # Volume change (if volume exists)
    if 'Volume' in df.columns:
        df['volume_change'] = df['Volume'].pct_change()
    
    # Stochastic Oscillator
    stoch_indicator = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
    df['STOCH_k'] = stoch_indicator.stoch()
    df['STOCH_d'] = stoch_indicator.stoch_signal()
    
    # Average Directional Index (ADX)
    adx_indicator = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    df['ADX'] = adx_indicator.adx()
    
    print(f"Added {len([c for c in df.columns if c not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Timestamp', 'target']])} technical indicators")
    
    return df


def add_candlestick_features(df):
    """
    Extract candlestick pattern features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLC data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with candlestick features added
    """
    print("Adding candlestick features...")
    
    df = df.copy()
    
    # Body size
    df['body_size'] = abs(df['Close'] - df['Open'])
    
    # Upper wick
    df['upper_wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    
    # Lower wick
    df['lower_wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    
    # Candle range
    df['candle_range'] = df['High'] - df['Low']
    
    # Is bullish
    df['is_bullish'] = (df['Close'] > df['Open']).astype(int)
    
    # Body to range ratio
    df['body_range_ratio'] = df['body_size'] / (df['candle_range'] + 1e-10)
    
    # Upper wick to body ratio
    df['upper_wick_ratio'] = df['upper_wick'] / (df['body_size'] + 1e-10)
    
    # Lower wick to body ratio
    df['lower_wick_ratio'] = df['lower_wick'] / (df['body_size'] + 1e-10)
    
    print(f"Added 8 candlestick features")
    
    return df


def add_lag_features(df, lags=[1, 2, 3, 5]):
    """
    Create lagged features for close prices and returns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with price data
    lags : list
        List of lag periods
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with lagged features added
    """
    print(f"Adding lag features for lags: {lags}...")
    
    df = df.copy()
    
    for lag in lags:
        df[f'close_lag_{lag}'] = df['Close'].shift(lag)
        df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        df[f'high_lag_{lag}'] = df['High'].shift(lag)
        df[f'low_lag_{lag}'] = df['Low'].shift(lag)
    
    print(f"Added {len(lags) * 4} lag features")
    
    return df


def add_rolling_features(df, windows=[5, 10, 20]):
    """
    Create rolling statistics features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with price data
    windows : list
        List of rolling window sizes
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with rolling features added
    """
    print(f"Adding rolling features for windows: {windows}...")
    
    df = df.copy()
    
    for window in windows:
        # Rolling mean
        df[f'rolling_mean_{window}'] = df['Close'].rolling(window=window).mean()
        
        # Rolling standard deviation
        df[f'rolling_std_{window}'] = df['Close'].rolling(window=window).std()
        
        # Rolling max
        df[f'rolling_max_{window}'] = df['Close'].rolling(window=window).max()
        
        # Rolling min
        df[f'rolling_min_{window}'] = df['Close'].rolling(window=window).min()
        
        # Rolling range
        df[f'rolling_range_{window}'] = df[f'rolling_max_{window}'] - df[f'rolling_min_{window}']
    
    print(f"Added {len(windows) * 5} rolling features")
    
    return df


def add_temporal_features(df):
    """
    Add time-based features from Timestamp.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with Timestamp column
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with temporal features added
    """
    print("Adding temporal features...")
    
    df = df.copy()
    
    # Hour of day
    df['hour'] = df['Timestamp'].dt.hour
    
    # Minute of hour
    df['minute'] = df['Timestamp'].dt.minute
    
    # Day of week (0=Monday, 6=Sunday)
    df['day_of_week'] = df['Timestamp'].dt.dayofweek
    
    # Is market open hour (assuming 9 AM to 3 PM)
    df['is_market_hours'] = ((df['hour'] >= 9) & (df['hour'] < 15)).astype(int)
    
    print(f"Added 4 temporal features")
    
    return df


def add_price_position_features(df):
    """
    Add features about price position relative to recent highs/lows.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with price data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with position features added
    """
    print("Adding price position features...")
    
    df = df.copy()
    
    # Distance from moving averages
    df['close_to_sma20'] = (df['Close'] - df['SMA_20']) / (df['SMA_20'] + 1e-10)
    df['close_to_ema20'] = (df['Close'] - df['EMA_20']) / (df['EMA_20'] + 1e-10)
    
    # Position within Bollinger Bands
    df['bb_position'] = (df['Close'] - df['BB_lower']) / (df['BB_width'] + 1e-10)
    
    # Distance from recent high/low
    df['distance_from_high_20'] = (df['rolling_max_20'] - df['Close']) / (df['Close'] + 1e-10)
    df['distance_from_low_20'] = (df['Close'] - df['rolling_min_20']) / (df['Close'] + 1e-10)
    
    print(f"Added 5 price position features")
    
    return df


def prepare_features(df):
    """
    Master function to apply all feature engineering steps.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with raw OHLC data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with all features added and NaN rows dropped
    """
    print("\n" + "="*60)
    print("FEATURE ENGINEERING PIPELINE")
    print("="*60)
    
    initial_rows = len(df)
    
    # Apply all feature engineering functions
    df = add_technical_indicators(df)
    df = add_candlestick_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_temporal_features(df)
    df = add_price_position_features(df)
    
    # Check NaN counts before dropping
    nan_counts = df.isnull().sum()
    print(f"\nNaN counts per column (top 10):")
    print(nan_counts[nan_counts > 0].sort_values(ascending=False).head(10))
    
    # Drop only rows where target is NaN (most important)
    if 'target' in df.columns:
        df = df[df['target'].notna()]
    
    # Fill remaining NaN values with forward fill, then backward fill, then 0
    df = df.ffill().bfill().fillna(0)
    
    final_rows = len(df)
    dropped_rows = initial_rows - final_rows
    
    print("\n" + "="*60)
    print(f"Feature engineering complete!")
    print(f"Initial rows: {initial_rows}")
    print(f"Final rows: {final_rows}")
    print(f"Dropped rows (NaN target): {dropped_rows}")
    print(f"Total features: {len(df.columns)}")
    print("="*60 + "\n")
    
    return df
