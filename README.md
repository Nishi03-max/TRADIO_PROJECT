# NIFTY Intraday Price Prediction using Machine Learning

This project predicts the direction of the next candle's closing price using historical NIFTY intraday data and machine learning models.

## 🎯 Objective

- Build ML models to predict if the next candle's close will be higher (1) or lower (0)
- Compare multiple models (Logistic Regression, Random Forest, XGBoost)
- Generate trading signals and calculate cumulative PnL
- Evaluate model performance using accuracy, precision, recall, and F1 score

## 📊 Dataset

- **Source**: 1 year of NIFTY intraday OHLC data
- **Expected Columns**: `Timestamp`, `Open`, `High`, `Low`, `Close`, `Volume` (optional)
- **Location**: Place your CSV file in `data/raw/nifty_intraday.csv`

## 🚀 Setup Instructions

### 1. Clone or Download the Project

```bash
cd NIFTY_MODEL
```

### 2. Create Virtual Environment (Recommended)

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Add Your Data

Place your NIFTY intraday CSV file in the `data/raw/` folder with the name `nifty_intraday.csv`.

**Required columns:**
- `Timestamp` - Date and time of the candle
- `Open` - Opening price
- `High` - Highest price
- `Low` - Lowest price
- `Close` - Closing price
- `Volume` - Trading volume (optional)

## 📁 Project Structure

```
NIFTY_MODEL/
├── data/
│   ├── raw/                    # Raw data files
│   │   └── nifty_intraday.csv  # Your input data (add this)
│   └── processed/              # Processed data and predictions
│       ├── train.csv
│       ├── test.csv
│       └── final_predictions.csv
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_experiments.ipynb
├── src/
│   ├── data_loader.py         # Data loading and preprocessing
│   ├── feature_engineering.py # Technical indicators and features
│   ├── models.py              # ML model training
│   ├── evaluation.py          # Model evaluation metrics
│   └── pnl_calculator.py      # Trading signals and PnL
├── models/                    # Saved trained models
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── scaler.pkl
├── results/                   # Plots and evaluation results
│   ├── model_comparison.csv
│   ├── pnl_curve.png
│   ├── confusion_matrix_*.png
│   ├── feature_importance_*.png
│   └── roc_curve_*.png
├── main.py                    # Main execution script
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md
```

## 🏃 Running the Project

### Option 1: Run the Complete Pipeline

Execute the main script to run the entire pipeline:

```powershell
python main.py
```

This will:
1. ✅ Load and preprocess data
2. ✅ Create target variable (1 if next close > current close, else 0)
3. ✅ Engineer 70+ technical features
4. ✅ Split data chronologically (70% train, 30% test)
5. ✅ Train 3 ML models (Logistic Regression, Random Forest, XGBoost)
6. ✅ Evaluate and compare models
7. ✅ Select best model
8. ✅ Generate trading signals
9. ✅ Calculate cumulative PnL
10. ✅ Save results and visualizations

### Option 2: Use Jupyter Notebooks

For interactive exploration:

```powershell
jupyter notebook
```

Then open:
- `01_eda.ipynb` - Exploratory Data Analysis
- `02_feature_engineering.ipynb` - Feature Engineering
- `03_model_experiments.ipynb` - Model Training and Experiments

## 📊 Features Generated

### Technical Indicators (20+)
- **Momentum**: RSI, MACD, Stochastic Oscillator
- **Trend**: SMA (5, 10, 20), EMA (5, 10, 20)
- **Volatility**: Bollinger Bands, ATR
- **Volume**: Volume change percentage
- **Others**: ADX (trend strength)

### Candlestick Features (8)
- Body size, upper/lower wicks, candle range
- Bullish/bearish indicator
- Body-to-range ratio, wick ratios

### Lag Features (16)
- Previous 1, 2, 3, 5 candle closes
- Previous returns, highs, lows

### Rolling Features (15)
- Rolling mean, std, max, min (windows: 5, 10, 20)
- Rolling range

### Temporal Features (4)
- Hour, minute, day of week
- Market hours indicator

### Price Position Features (5)
- Distance from moving averages
- Bollinger Band position
- Distance from recent highs/lows

**Total: 70+ Features**

## 🤖 Models Used

1. **Logistic Regression** - Baseline linear model
2. **Random Forest** - Ensemble tree-based model
3. **XGBoost** - Gradient boosting (typically best performer)

## 📈 Evaluation Metrics

- **Accuracy** - Overall correctness
- **Precision** - Accuracy of positive predictions
- **Recall** - Coverage of actual positives
- **F1 Score** - Harmonic mean of precision and recall
- **Confusion Matrix** - Visual breakdown of predictions
- **ROC Curve & AUC** - Model discrimination ability
- **Feature Importance** - Most influential features

## 💰 PnL Calculation Strategy

**Trading Logic:**
- **Buy Signal (Prediction = 1)**: Take long position at current close, exit at next close
  - PnL = Next Close - Current Close
- **Sell Signal (Prediction = 0)**: Take short position at current close, exit at next close
  - PnL = Current Close - Next Close

**PnL Metrics:**
- Cumulative PnL over test period
- Win rate (% of profitable trades)
- Average win/loss
- Maximum drawdown

## 📤 Output Files

After running `main.py`, you'll get:

### 1. Final Predictions CSV
**File**: `data/processed/final_predictions.csv`

Columns:
- `Timestamp` - Time of prediction
- `Close` - Actual closing price
- `Predicted` - Model prediction (0 or 1)
- `model_call` - Trading signal ('buy' or 'sell')
- `model_pnl` - Cumulative PnL

### 2. Model Comparison
**File**: `results/model_comparison.csv`
- Comparison of all models with metrics

### 3. Visualizations
- `pnl_curve.png` - Cumulative PnL over time
- `trade_distribution.png` - Distribution of individual trades
- `confusion_matrix_*.png` - Confusion matrices for each model
- `feature_importance_*.png` - Top features for tree models
- `roc_curve_*.png` - ROC curves for all models

### 4. Saved Models
All trained models saved in `models/` folder:
- `logistic_regression.pkl`
- `random_forest.pkl`
- `xgboost.pkl`
- `scaler.pkl`

## 🔧 Customization

### Change Train-Test Split Ratio

In `main.py`, modify:
```python
X_train, X_test, y_train, y_test, test_df = train_test_split_timeseries(df, train_ratio=0.7)
```

### Add More Features

Edit `src/feature_engineering.py` to add custom features.

### Tune Model Parameters

Edit `src/models.py` to adjust hyperparameters:
```python
model = XGBClassifier(
    n_estimators=200,      # More trees
    max_depth=8,           # Deeper trees
    learning_rate=0.05,    # Slower learning
    # ... other parameters
)
```

### Change Data Path

In `main.py`:
```python
data_path = 'data/raw/your_custom_file.csv'
```

## 📝 Example Usage

```python
# Load a saved model and make predictions
from src.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

# Load model and scaler
model = load_model('xgboost')
scaler = joblib.load('models/scaler.pkl')

# Prepare your new data (with same features)
# X_new = ... your feature-engineered data

# Scale and predict
X_new_scaled = scaler.transform(X_new)
predictions = model.predict(X_new_scaled)
```

## ⚠️ Important Notes

1. **Time-Series Split**: The data is split chronologically (no shuffling) to prevent look-ahead bias
2. **Feature Engineering**: NaN rows are dropped after indicator calculation (~20-50 rows)
3. **Scaling**: Always scale test data using the scaler fitted on training data only
4. **Target Variable**: Last row is dropped (no next candle to predict)

## 🐛 Troubleshooting

### Missing Data File
```
❌ ERROR: Data file not found at data/raw/nifty_intraday.csv
```
**Solution**: Place your CSV file in `data/raw/` folder with correct name.

### Missing Columns
```
❌ ERROR: Required columns not found
```
**Solution**: Ensure CSV has: Timestamp, Open, High, Low, Close

### Import Errors
```
❌ ERROR: No module named 'pandas_ta'
```
**Solution**: Install dependencies: `pip install -r requirements.txt`

### Memory Issues
If dataset is very large, reduce features or use sampling:
```python
df = df.sample(frac=0.5, random_state=42)  # Use 50% of data
```

## 📚 Dependencies

- pandas 2.0.3
- numpy 1.24.3
- scikit-learn 1.3.0
- xgboost 2.0.0
- matplotlib 3.7.1
- seaborn 0.12.2
- pandas-ta 0.3.14b
- joblib 1.3.1
- ta 0.11.0

## 📊 Performance Tips

1. **More data is better** - At least 6 months recommended
2. **Feature selection** - Remove low-importance features for faster training
3. **Hyperparameter tuning** - Use GridSearchCV for optimal parameters
4. **Ensemble methods** - Combine predictions from multiple models
5. **Cross-validation** - Use time-series cross-validation for robust evaluation

## 🤝 Contributing

Feel free to:
- Add more technical indicators
- Implement additional models (Neural Networks, SVM, etc.)
- Enhance PnL calculation with transaction costs
- Add risk management features (stop-loss, position sizing)

## 📄 License

This project is for educational purposes. Use at your own risk. Not financial advice.

## 🎓 Learning Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [pandas-ta Documentation](https://github.com/twopirllc/pandas-ta)
- [Technical Analysis Library](https://technical-analysis-library-in-python.readthedocs.io/)

## 📧 Contact

For questions or issues, please open an issue in the repository.

---

**Happy Trading! 🚀📈**

*Remember: Past performance does not guarantee future results. Always practice proper risk management.*
