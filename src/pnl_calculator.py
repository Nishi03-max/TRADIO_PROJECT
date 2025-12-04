"""
PnL calculation module for trading signals and profit/loss tracking.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def generate_signals(model, test_df, X_test):
    """
    Generate trading signals using the model predictions.
    
    Parameters:
    -----------
    model : sklearn/xgboost model
        Trained model
    test_df : pd.DataFrame
        Test dataframe with Timestamp and OHLC data
    X_test : array-like
        Test features
        
    Returns:
    --------
    pd.DataFrame
        Test dataframe with 'Predicted' and 'model_call' columns added
    """
    print("\n" + "="*60)
    print("GENERATING TRADING SIGNALS")
    print("="*60)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Create a copy to avoid SettingWithCopyWarning
    test_df = test_df.copy()
    
    # Add predictions
    test_df['Predicted'] = predictions
    
    # Add model_call column
    test_df['model_call'] = test_df['Predicted'].apply(
        lambda x: 'buy' if x == 1 else 'sell'
    )
    
    # Print signal distribution
    signal_counts = test_df['model_call'].value_counts()
    print(f"\nSignal Distribution:")
    print(signal_counts)
    print(f"\nBuy signals: {signal_counts.get('buy', 0)} ({signal_counts.get('buy', 0)/len(test_df)*100:.2f}%)")
    print(f"Sell signals: {signal_counts.get('sell', 0)} ({signal_counts.get('sell', 0)/len(test_df)*100:.2f}%)")
    
    return test_df


def calculate_pnl(test_df):
    """
    Calculate cumulative PnL based on trading signals.
    
    Strategy:
    - For each candle, take a position based on the prediction
    - Close the position at the next candle's close price
    - Buy signal (1): Long position - profit if next close > current close
    - Sell signal (0): Short position - profit if next close < current close
    
    Parameters:
    -----------
    test_df : pd.DataFrame
        Test dataframe with 'Predicted' and 'Close' columns
        
    Returns:
    --------
    pd.DataFrame
        Test dataframe with 'model_pnl' column added
    """
    print("\n" + "="*60)
    print("CALCULATING PnL")
    print("="*60)
    
    test_df = test_df.copy()
    
    # Initialize PnL tracking
    model_pnl = 0
    pnl_list = []
    trade_pnl_list = []
    
    # Iterate through each row
    for i in range(len(test_df)):
        current_close = test_df.iloc[i]['Close']
        prediction = test_df.iloc[i]['Predicted']
        
        # Check if next candle exists
        if i + 1 < len(test_df):
            next_close = test_df.iloc[i + 1]['Close']
            
            # Calculate PnL change for this trade
            if prediction == 1:  # Buy signal (Long position)
                pnl_change = next_close - current_close
            else:  # Sell signal (Short position)
                pnl_change = current_close - next_close
            
            # Update cumulative PnL
            model_pnl += pnl_change
            trade_pnl_list.append(pnl_change)
        else:
            # Last candle - no next candle to close position
            trade_pnl_list.append(0)
        
        # Append cumulative PnL
        pnl_list.append(model_pnl)
    
    # Add PnL column to dataframe
    test_df['model_pnl'] = pnl_list
    test_df['trade_pnl'] = trade_pnl_list
    
    # Calculate statistics
    final_pnl = test_df['model_pnl'].iloc[-1]
    total_trades = len(test_df) - 1  # Exclude last candle (no trade)
    winning_trades = len([x for x in trade_pnl_list if x > 0])
    losing_trades = len([x for x in trade_pnl_list if x < 0])
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
    
    avg_win = np.mean([x for x in trade_pnl_list if x > 0]) if winning_trades > 0 else 0
    avg_loss = np.mean([x for x in trade_pnl_list if x < 0]) if losing_trades > 0 else 0
    
    print(f"\nPnL STATISTICS:")
    print(f"  Final PnL: {final_pnl:.2f}")
    print(f"  Total Trades: {total_trades}")
    print(f"  Winning Trades: {winning_trades}")
    print(f"  Losing Trades: {losing_trades}")
    print(f"  Win Rate: {win_rate:.2f}%")
    print(f"  Average Win: {avg_win:.2f}")
    print(f"  Average Loss: {avg_loss:.2f}")
    if avg_loss != 0:
        print(f"  Profit Factor: {abs(avg_win/avg_loss):.2f}")
    
    # Maximum drawdown
    cumulative_max = test_df['model_pnl'].cummax()
    drawdown = test_df['model_pnl'] - cumulative_max
    max_drawdown = drawdown.min()
    print(f"  Maximum Drawdown: {max_drawdown:.2f}")
    
    return test_df


def plot_pnl_curve(test_df, output_dir='results'):
    """
    Plot cumulative PnL curve over time.
    
    Parameters:
    -----------
    test_df : pd.DataFrame
        Test dataframe with 'Timestamp' and 'model_pnl' columns
    output_dir : str
        Directory to save the plot
    """
    print("\nGenerating PnL curve plot...")
    
    # Create figure
    plt.figure(figsize=(14, 6))
    plt.plot(test_df['Timestamp'], test_df['model_pnl'], linewidth=2, color='blue')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    plt.xlabel('Timestamp')
    plt.ylabel('Cumulative PnL')
    plt.title('Cumulative Profit & Loss Over Time')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Add final PnL annotation
    final_pnl = test_df['model_pnl'].iloc[-1]
    plt.text(0.02, 0.98, f'Final PnL: {final_pnl:.2f}', 
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=12, fontweight='bold')
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'pnl_curve.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"PnL curve saved to: {filepath}")


def plot_trade_distribution(test_df, output_dir='results'):
    """
    Plot distribution of individual trade PnL.
    
    Parameters:
    -----------
    test_df : pd.DataFrame
        Test dataframe with 'trade_pnl' column
    output_dir : str
        Directory to save the plot
    """
    print("\nGenerating trade distribution plot...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(test_df['trade_pnl'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Trade PnL')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Trade PnL')
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1].boxplot(test_df['trade_pnl'], vert=True)
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1].set_ylabel('Trade PnL')
    axes[1].set_title('Trade PnL Box Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'trade_distribution.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Trade distribution plot saved to: {filepath}")


def save_final_predictions(test_df, output_path='data/processed/final_predictions.csv'):
    """
    Save final predictions with required columns.
    
    Parameters:
    -----------
    test_df : pd.DataFrame
        Test dataframe with all columns
    output_path : str
        Path to save the CSV file
    """
    print("\n" + "="*60)
    print("SAVING FINAL PREDICTIONS")
    print("="*60)
    
    # Select required columns
    required_columns = ['Timestamp', 'Close', 'Predicted', 'model_call', 'model_pnl']
    
    # Check if all columns exist
    missing_cols = [col for col in required_columns if col not in test_df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in test_df: {missing_cols}")
    
    # Create final output
    final_output = test_df[required_columns].copy()
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_output.to_csv(output_path, index=False)
    
    print(f"\nFinal predictions saved to: {output_path}")
    print(f"Total records: {len(final_output)}")
    print(f"Columns: {list(final_output.columns)}")
    print(f"\nFirst few rows:")
    print(final_output.head())
    print(f"\nLast few rows:")
    print(final_output.tail())
