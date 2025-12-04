"""
Main execution script for NIFTY Intraday Price Prediction ML Project.

This script runs the entire pipeline:
1. Load data
2. Create target variable
3. Feature engineering
4. Train-test split
5. Feature scaling
6. Train models
7. Evaluate and compare models
8. Generate trading signals
9. Calculate PnL
10. Save results
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import load_data, create_target_column, train_test_split_timeseries, save_processed_data
from src.feature_engineering import prepare_features
from src.models import train_all_models, save_model
from src.evaluation import compare_models, plot_confusion_matrix, plot_feature_importance, plot_roc_curve
from src.pnl_calculator import generate_signals, calculate_pnl, plot_pnl_curve, plot_trade_distribution, save_final_predictions

from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd


def main():
    """
    Main execution function for the entire ML pipeline.
    """
    print("\n" + "="*70)
    print(" "*15 + "NIFTY INTRADAY PRICE PREDICTION")
    print(" "*20 + "ML PROJECT PIPELINE")
    print("="*70 + "\n")
    
    # ========================================================================
    # STEP 1: LOAD DATA
    # ========================================================================
    print("\n📂 STEP 1: Loading Data...")
    print("-" * 70)
    
    data_path = 'data/raw/nifty_intraday.csv'
    
    if not os.path.exists(data_path):
        print(f"❌ ERROR: Data file not found at {data_path}")
        print(f"Please place your NIFTY intraday OHLC CSV file in the data/raw/ folder")
        print(f"Expected columns: Timestamp, Open, High, Low, Close")
        return
    
    df = load_data(data_path)
    
    # ========================================================================
    # STEP 2: CREATE TARGET VARIABLE
    # ========================================================================
    print("\n🎯 STEP 2: Creating Target Variable...")
    print("-" * 70)
    
    df = create_target_column(df)
    
    # ========================================================================
    # STEP 3: FEATURE ENGINEERING
    # ========================================================================
    print("\n🔧 STEP 3: Feature Engineering...")
    print("-" * 70)
    
    df = prepare_features(df)
    
    # ========================================================================
    # STEP 4: TRAIN-TEST SPLIT
    # ========================================================================
    print("\n✂️ STEP 4: Train-Test Split...")
    print("-" * 70)
    
    X_train, X_test, y_train, y_test, test_df = train_test_split_timeseries(df, train_ratio=0.7)
    
    # Save feature names for later use
    feature_names = X_train.columns.tolist()
    
    # ========================================================================
    # STEP 5: FEATURE SCALING
    # ========================================================================
    print("\n📊 STEP 5: Feature Scaling...")
    print("-" * 70)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    scaler_path = 'models/scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")
    
    # ========================================================================
    # STEP 6: TRAIN MODELS
    # ========================================================================
    print("\n🤖 STEP 6: Training Models...")
    print("-" * 70)
    
    models = train_all_models(X_train_scaled, y_train)
    
    # ========================================================================
    # STEP 7: SAVE MODELS
    # ========================================================================
    print("\n💾 STEP 7: Saving Models...")
    print("-" * 70)
    
    save_model(models['Logistic Regression'], 'logistic_regression')
    save_model(models['Random Forest'], 'random_forest')
    save_model(models['XGBoost'], 'xgboost')
    
    # ========================================================================
    # STEP 8: EVALUATE AND COMPARE MODELS
    # ========================================================================
    print("\n📈 STEP 8: Evaluating and Comparing Models...")
    print("-" * 70)
    
    best_model_name = compare_models(models, X_test_scaled, y_test)
    best_model = models[best_model_name]
    
    # ========================================================================
    # STEP 9: GENERATE VISUALIZATIONS
    # ========================================================================
    print("\n📊 STEP 9: Generating Visualizations...")
    print("-" * 70)
    
    # Confusion matrices for all models
    for model_name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        plot_confusion_matrix(y_test, y_pred, model_name)
    
    # Feature importance for tree-based models
    for model_name, model in models.items():
        if model_name in ['Random Forest', 'XGBoost']:
            plot_feature_importance(model, feature_names, model_name, top_n=20)
    
    # ROC curves
    for model_name, model in models.items():
        plot_roc_curve(model, X_test_scaled, y_test, model_name)
    
    # ========================================================================
    # STEP 10: GENERATE TRADING SIGNALS
    # ========================================================================
    print("\n💡 STEP 10: Generating Trading Signals...")
    print("-" * 70)
    
    test_df = generate_signals(best_model, test_df, X_test_scaled)
    
    # ========================================================================
    # STEP 11: CALCULATE PnL
    # ========================================================================
    print("\n💰 STEP 11: Calculating PnL...")
    print("-" * 70)
    
    test_df = calculate_pnl(test_df)
    
    # ========================================================================
    # STEP 12: PLOT PnL RESULTS
    # ========================================================================
    print("\n📉 STEP 12: Plotting PnL Results...")
    print("-" * 70)
    
    plot_pnl_curve(test_df)
    plot_trade_distribution(test_df)
    
    # ========================================================================
    # STEP 13: SAVE FINAL OUTPUT
    # ========================================================================
    print("\n💾 STEP 13: Saving Final Output...")
    print("-" * 70)
    
    save_final_predictions(test_df, 'data/processed/final_predictions.csv')
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print(" "*25 + "🎉 PIPELINE COMPLETE! 🎉")
    print("="*70)
    
    final_pnl = test_df['model_pnl'].iloc[-1]
    total_trades = len(test_df) - 1
    winning_trades = len([x for x in test_df['trade_pnl'] if x > 0])
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
    
    print(f"\n📊 SUMMARY:")
    print(f"  🏆 Best Model: {best_model_name}")
    print(f"  💰 Final PnL: {final_pnl:.2f}")
    print(f"  📈 Total Trades: {total_trades}")
    print(f"  ✅ Winning Trades: {winning_trades}")
    print(f"  📊 Win Rate: {win_rate:.2f}%")
    
    print(f"\n📁 OUTPUT FILES:")
    print(f"  • Final Predictions: data/processed/final_predictions.csv")
    print(f"  • Model Comparison: results/model_comparison.csv")
    print(f"  • PnL Curve: results/pnl_curve.png")
    print(f"  • Trade Distribution: results/trade_distribution.png")
    print(f"  • Confusion Matrices: results/confusion_matrix_*.png")
    print(f"  • Feature Importance: results/feature_importance_*.png")
    print(f"  • ROC Curves: results/roc_curve_*.png")
    print(f"  • Saved Models: models/*.pkl")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
