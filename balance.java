<?xml version="1.0" encoding="UTF-8"?>
<project>
    <metadata>
        <name>NIFTY Intraday Price Prediction ML Project</name>
        <objective>Build ML models to predict next candle direction and calculate PnL</objective>
        <dataset>1 year NIFTY intraday OHLC data in data/ folder</dataset>
    </metadata>

    <!-- PHASE 1: PROJECT SETUP -->
    <phase id="1" name="Project Setup and Structure">
        <task id="1.1" priority="high">
            <name>Create Project Directory Structure</name>
            <description>Set up the folder hierarchy for the project</description>
            <structure>
                nifty-price-prediction/
                ├── data/
                │   ├── raw/
                │   └── processed/
                ├── notebooks/
                ├── src/
                ├── models/
                ├── results/
                ├── main.py
                ├── requirements.txt
                ├── README.md
                └── .gitignore
            </structure>
        </task>

        <task id="1.2" priority="high">
            <name>Create requirements.txt</name>
            <description>List all Python dependencies</description>
            <dependencies>
                <package>pandas==2.0.3</package>
                <package>numpy==1.24.3</package>
                <package>scikit-learn==1.3.0</package>
                <package>xgboost==2.0.0</package>
                <package>matplotlib==3.7.1</package>
                <package>seaborn==0.12.2</package>
                <package>pandas-ta==0.3.14b</package>
                <package>joblib==1.3.1</package>
                <package>ta==0.11.0</package>
            </dependencies>
        </task>

        <task id="1.3" priority="medium">
            <name>Create .gitignore</name>
            <description>Exclude unnecessary files from git</description>
            <content>
                __pycache__/
                *.pyc
                .ipynb_checkpoints/
                *.pkl
                .DS_Store
                .vscode/
                *.log
            </content>
        </task>

        <task id="1.4" priority="high">
            <name>Initialize Git Repository</name>
            <description>Set up version control</description>
            <commands>
                <command>git init</command>
                <command>git add .</command>
                <command>git commit -m "Initial project setup"</command>
                <command>git remote add origin [your-repo-url]</command>
                <command>git push -u origin main</command>
            </commands>
        </task>
    </phase>

    <!-- PHASE 2: DATA LOADING AND EXPLORATION -->
    <phase id="2" name="Data Loading and EDA">
        <task id="2.1" priority="high">
            <name>Create data_loader.py</name>
            <description>Module to load and preprocess raw data</description>
            <filepath>src/data_loader.py</filepath>
            <functions>
                <function>
                    <name>load_data(filepath)</name>
                    <purpose>Load CSV from data/raw/ folder</purpose>
                    <steps>
                        <step>Read CSV using pandas</step>
                        <step>Parse Timestamp column as datetime</step>
                        <step>Sort by Timestamp in ascending order</step>
                        <step>Reset index</step>
                        <step>Check for missing values</step>
                        <step>Return cleaned dataframe</step>
                    </steps>
                </function>
                <function>
                    <name>create_target_column(df)</name>
                    <purpose>Generate target variable</purpose>
                    <steps>
                        <step>Create new column: next_close = df['Close'].shift(-1)</step>
                        <step>Create target: df['target'] = (next_close > df['Close']).astype(int)</step>
                        <step>Drop last row (no target available)</step>
                        <step>Drop next_close helper column</step>
                        <step>Return dataframe with target column</step>
                    </steps>
                </function>
            </functions>
        </task>

        <task id="2.2" priority="medium">
            <name>Create EDA Notebook</name>
            <description>Exploratory data analysis</description>
            <filepath>notebooks/01_eda.ipynb</filepath>
            <analysis>
                <item>Load data and display first/last rows</item>
                <item>Check data shape and info</item>
                <item>Statistical summary (describe)</item>
                <item>Check for missing values</item>
                <item>Plot Close price over time</item>
                <item>Analyze target distribution (class balance)</item>
                <item>Plot OHLC candlestick chart (sample period)</item>
                <item>Check for any data anomalies</item>
            </analysis>
        </task>
    </phase>

    <!-- PHASE 3: FEATURE ENGINEERING -->
    <phase id="3" name="Feature Engineering">
        <task id="3.1" priority="high">
            <name>Create feature_engineering.py</name>
            <description>Generate features for ML models</description>
            <filepath>src/feature_engineering.py</filepath>
            <functions>
                <function>
                    <name>add_technical_indicators(df)</name>
                    <purpose>Calculate technical indicators using pandas_ta</purpose>
                    <features>
                        <feature>
                            <name>returns</name>
                            <formula>df['Close'].pct_change()</formula>
                        </feature>
                        <feature>
                            <name>RSI_14</name>
                            <description>Relative Strength Index with 14 period</description>
                        </feature>
                        <feature>
                            <name>MACD</name>
                            <description>MACD line, signal, histogram</description>
                        </feature>
                        <feature>
                            <name>BB_upper, BB_lower</name>
                            <description>Bollinger Bands (20 period, 2 std)</description>
                        </feature>
                        <feature>
                            <name>SMA_5, SMA_10, SMA_20</name>
                            <description>Simple Moving Averages</description>
                        </feature>
                        <feature>
                            <name>EMA_5, EMA_10, EMA_20</name>
                            <description>Exponential Moving Averages</description>
                        </feature>
                        <feature>
                            <name>ATR_14</name>
                            <description>Average True Range (volatility)</description>
                        </feature>
                        <feature>
                            <name>volume_change</name>
                            <formula>df['Volume'].pct_change() if volume exists</formula>
                        </feature>
                    </features>
                </function>
                <function>
                    <name>add_candlestick_features(df)</name>
                    <purpose>Extract candlestick patterns</purpose>
                    <features>
                        <feature>
                            <name>body_size</name>
                            <formula>abs(Close - Open)</formula>
                        </feature>
                        <feature>
                            <name>upper_wick</name>
                            <formula>High - max(Open, Close)</formula>
                        </feature>
                        <feature>
                            <name>lower_wick</name>
                            <formula>min(Open, Close) - Low</formula>
                        </feature>
                        <feature>
                            <name>candle_range</name>
                            <formula>High - Low</formula>
                        </feature>
                        <feature>
                            <name>is_bullish</name>
                            <formula>(Close > Open).astype(int)</formula>
                        </feature>
                    </features>
                </function>
                <function>
                    <name>add_lag_features(df, lags=[1,2,3,5])</name>
                    <purpose>Create lagged features</purpose>
                    <features>
                        <feature>
                            <name>close_lag_1, close_lag_2, etc.</name>
                            <description>Previous candle closes</description>
                        </feature>
                        <feature>
                            <name>returns_lag_1, returns_lag_2, etc.</name>
                            <description>Previous returns</description>
                        </feature>
                    </features>
                </function>
                <function>
                    <name>add_rolling_features(df, windows=[5,10,20])</name>
                    <purpose>Rolling statistics</purpose>
                    <features>
                        <feature>
                            <name>rolling_mean_5, rolling_mean_10, etc.</name>
                            <description>Rolling mean of close prices</description>
                        </feature>
                        <feature>
                            <name>rolling_std_5, rolling_std_10, etc.</name>
                            <description>Rolling standard deviation</description>
                        </feature>
                        <feature>
                            <name>rolling_max_5, rolling_min_5, etc.</name>
                            <description>Rolling max/min</description>
                        </feature>
                    </features>
                </function>
                <function>
                    <name>add_temporal_features(df)</name>
                    <purpose>Time-based features</purpose>
                    <features>
                        <feature>
                            <name>hour</name>
                            <formula>df['Timestamp'].dt.hour</formula>
                        </feature>
                        <feature>
                            <name>day_of_week</name>
                            <formula>df['Timestamp'].dt.dayofweek</formula>
                        </feature>
                        <feature>
                            <name>minute</name>
                            <formula>df['Timestamp'].dt.minute</formula>
                        </feature>
                    </features>
                </function>
                <function>
                    <name>prepare_features(df)</name>
                    <purpose>Master function that calls all feature engineering</purpose>
                    <steps>
                        <step>Call add_technical_indicators(df)</step>
                        <step>Call add_candlestick_features(df)</step>
                        <step>Call add_lag_features(df)</step>
                        <step>Call add_rolling_features(df)</step>
                        <step>Call add_temporal_features(df)</step>
                        <step>Drop rows with NaN (from indicators/lags)</step>
                        <step>Return cleaned dataframe</step>
                    </steps>
                </function>
            </functions>
        </task>

        <task id="3.2" priority="medium">
            <name>Create Feature Engineering Notebook</name>
            <description>Test and visualize features</description>
            <filepath>notebooks/02_feature_engineering.ipynb</filepath>
            <content>
                <item>Load data using data_loader</item>
                <item>Apply feature engineering functions</item>
                <item>Check for NaN values after feature creation</item>
                <item>Display correlation matrix</item>
                <item>Plot feature importance using Random Forest</item>
                <item>Visualize some technical indicators on chart</item>
            </content>
        </task>
    </phase>

    <!-- PHASE 4: DATA SPLITTING -->
    <phase id="4" name="Train-Test Split">
        <task id="4.1" priority="high">
            <name>Implement time-based split in data_loader.py</name>
            <description>Split data chronologically (no shuffle)</description>
            <function>
                <name>train_test_split_timeseries(df, train_ratio=0.7)</name>
                <steps>
                    <step>Calculate split index: int(len(df) * train_ratio)</step>
                    <step>Split: train = df[:split_index]</step>
                    <step>Split: test = df[split_index:]</step>
                    <step>Separate features (X) and target (y)</step>
                    <step>X_train, y_train = train.drop(['target', 'Timestamp'], axis=1), train['target']</step>
                    <step>X_test, y_test = test.drop(['target', 'Timestamp'], axis=1), test['target']</step>
                    <step>Return X_train, X_test, y_train, y_test, test (with Timestamp)</step>
                </steps>
            </function>
        </task>

        <task id="4.2" priority="high">
            <name>Feature Scaling</name>
            <description>Standardize features</description>
            <implementation>
                <step>Import StandardScaler from sklearn</step>
                <step>Fit scaler on X_train only</step>
                <step>Transform both X_train and X_test</step>
                <step>Save scaler to models/ folder using joblib</step>
            </implementation>
        </task>

        <task id="4.3" priority="medium">
            <name>Save processed data</name>
            <description>Save train/test sets for reproducibility</description>
            <steps>
                <step>Save train data to data/processed/train.csv</step>
                <step>Save test data to data/processed/test.csv</step>
            </steps>
        </task>
    </phase>

    <!-- PHASE 5: MODEL TRAINING -->
    <phase id="5" name="Model Building and Training">
        <task id="5.1" priority="high">
            <name>Create models.py</name>
            <description>Define and train ML models</description>
            <filepath>src/models.py</filepath>
            <models>
                <model id="1">
                    <name>Logistic Regression</name>
                    <library>sklearn.linear_model.LogisticRegression</library>
                    <parameters>
                        <param>max_iter=1000</param>
                        <param>random_state=42</param>
                    </parameters>
                    <purpose>Baseline linear model</purpose>
                </model>
                <model id="2">
                    <name>Random Forest Classifier</name>
                    <library>sklearn.ensemble.RandomForestClassifier</library>
                    <parameters>
                        <param>n_estimators=100</param>
                        <param>max_depth=10</param>
                        <param>random_state=42</param>
                        <param>n_jobs=-1</param>
                    </parameters>
                    <purpose>Ensemble tree-based model</purpose>
                </model>
                <model id="3">
                    <name>XGBoost Classifier</name>
                    <library>xgboost.XGBClassifier</library>
                    <parameters>
                        <param>n_estimators=100</param>
                        <param>max_depth=6</param>
                        <param>learning_rate=0.1</param>
                        <param>random_state=42</param>
                        <param>eval_metric='logloss'</param>
                    </parameters>
                    <purpose>Gradient boosting model (recommended)</purpose>
                </model>
            </models>
            <functions>
                <function>
                    <name>train_logistic_regression(X_train, y_train)</name>
                    <steps>
                        <step>Initialize LogisticRegression</step>
                        <step>Fit on X_train, y_train</step>
                        <step>Return trained model</step>
                    </steps>
                </function>
                <function>
                    <name>train_random_forest(X_train, y_train)</name>
                    <steps>
                        <step>Initialize RandomForestClassifier</step>
                        <step>Fit on X_train, y_train</step>
                        <step>Return trained model</step>
                    </steps>
                </function>
                <function>
                    <name>train_xgboost(X_train, y_train)</name>
                    <steps>
                        <step>Initialize XGBClassifier</step>
                        <step>Fit on X_train, y_train</step>
                        <step>Return trained model</step>
                    </steps>
                </function>
                <function>
                    <name>save_model(model, filename)</name>
                    <steps>
                        <step>Use joblib.dump(model, f'models/{filename}.pkl')</step>
                    </steps>
                </function>
                <function>
                    <name>load_model(filename)</name>
                    <steps>
                        <step>Use joblib.load(f'models/{filename}.pkl')</step>
                        <step>Return loaded model</step>
                    </steps>
                </function>
            </functions>
        </task>

        <task id="5.2" priority="medium">
            <name>Create Model Experiments Notebook</name>
            <description>Train and compare models interactively</description>
            <filepath>notebooks/03_model_experiments.ipynb</filepath>
            <steps>
                <step>Load processed train/test data</step>
                <step>Train all three models</step>
                <step>Make predictions on test set</step>
                <step>Compare accuracies</step>
                <step>Visualize confusion matrices</step>
                <step>Plot feature importance for tree-based models</step>
            </steps>
        </task>
    </phase>

    <!-- PHASE 6: MODEL EVALUATION -->
    <phase id="6" name="Model Evaluation">
        <task id="6.1" priority="high">
            <name>Create evaluation.py</name>
            <description>Evaluate model performance</description>
            <filepath>src/evaluation.py</filepath>
            <functions>
                <function>
                    <name>evaluate_model(model, X_test, y_test, model_name)</name>
                    <metrics>
                        <metric>
                            <name>Accuracy</name>
                            <code>accuracy_score(y_test, y_pred)</code>
                        </metric>
                        <metric>
                            <name>Precision</name>
                            <code>precision_score(y_test, y_pred)</code>
                        </metric>
                        <metric>
                            <name>Recall</name>
                            <code>recall_score(y_test, y_pred)</code>
                        </metric>
                        <metric>
                            <name>F1 Score</name>
                            <code>f1_score(y_test, y_pred)</code>
                        </metric>
                    </metrics>
                    <steps>
                        <step>Make predictions: y_pred = model.predict(X_test)</step>
                        <step>Calculate all metrics</step>
                        <step>Print results</step>
                        <step>Return dictionary of metrics</step>
                    </steps>
                </function>
                <function>
                    <name>compare_models(models_dict, X_test, y_test)</name>
                    <purpose>Compare multiple models and select best</purpose>
                    <steps>
                        <step>Loop through all models</step>
                        <step>Evaluate each model</step>
                        <step>Store results in list</step>
                        <step>Create DataFrame for comparison</step>
                        <step>Save to results/model_comparison.csv</step>
                        <step>Return best model name based on accuracy</step>
                    </steps>
                </function>
                <function>
                    <name>plot_confusion_matrix(y_test, y_pred, model_name)</name>
                    <steps>
                        <step>Calculate confusion matrix</step>
                        <step>Plot using seaborn heatmap</step>
                        <step>Save to results/confusion_matrix_{model_name}.png</step>
                    </steps>
                </function>
                <function>
                    <name>plot_feature_importance(model, feature_names, model_name)</name>
                    <steps>
                        <step>Extract feature importances (tree-based models)</step>
                        <step>Create horizontal bar plot</step>
                        <step>Show top 20 features</step>
                        <step>Save to results/feature_importance_{model_name}.png</step>
                    </steps>
                </function>
            </functions>
        </task>
    </phase>

    <!-- PHASE 7: SIGNAL GENERATION AND PNL -->
    <phase id="7" name="Signal Generation and PnL Calculation">
        <task id="7.1" priority="high">
            <name>Create pnl_calculator.py</name>
            <description>Generate trading signals and calculate PnL</description>
            <filepath>src/pnl_calculator.py</filepath>
            <functions>
                <function>
                    <name>generate_signals(model, test_df, X_test)</name>
                    <purpose>Add model_call column to test dataframe</purpose>
                    <steps>
                        <step>Make predictions: predictions = model.predict(X_test)</step>
                        <step>Create copy of test_df to avoid warnings</step>
                        <step>Add column: test_df['Predicted'] = predictions</step>
                        <step>Add column: test_df['model_call'] = ['buy' if p == 1 else 'sell' for p in predictions]</step>
                        <step>Return test_df with new columns</step>
                    </steps>
                </function>
                <function>
                    <name>calculate_pnl(test_df)</name>
                    <purpose>Calculate cumulative PnL based on signals</purpose>
                    <logic>
                        <description>Strategy: For each candle, take position based on prediction, close at next candle</description>
                        <algorithm>
                            Initialize: model_pnl = 0
                            
                            For each row i in test_df:
                                current_close = row['Close']
                                prediction = row['Predicted']
                                
                                if i+1 &lt; len(test_df):  # If next candle exists
                                    next_close = test_df.iloc[i+1]['Close']
                                    
                                    if prediction == 1:  # Buy signal
                                        pnl_change = next_close - current_close
                                        model_pnl += pnl_change
                                    else:  # Sell signal (short)
                                        pnl_change = current_close - next_close
                                        model_pnl += pnl_change
                                
                                test_df.at[i, 'model_pnl'] = model_pnl
                        </algorithm>
                    </logic>
                    <steps>
                        <step>Initialize model_pnl = 0</step>
                        <step>Create empty list for pnl values</step>
                        <step>Iterate through test_df rows</step>
                        <step>Calculate PnL change for each trade</step>
                        <step>Update cumulative model_pnl</step>
                        <step>Append to list</step>
                        <step>Add model_pnl column to test_df</step>
                        <step>Return test_df with model_pnl</step>
                    </steps>
                </function>
                <function>
                    <name>plot_pnl_curve(test_df)</name>
                    <steps>
                        <step>Plot Timestamp vs model_pnl</step>
                        <step>Add title and labels</step>
                        <step>Save to results/pnl_curve.png</step>
                    </steps>
                </function>
            </functions>
        </task>

        <task id="7.2" priority="high">
            <name>Generate final predictions CSV</name>
            <description>Create output file with all required columns</description>
            <steps>
                <step>Select required columns: Timestamp, Close, Predicted, model_call, model_pnl</step>
                <step>Save to data/processed/final_predictions.csv</step>
                <step>Verify file contains all expected columns</step>
            </steps>
        </task>
    </phase>

    <!-- PHASE 8: MAIN EXECUTION SCRIPT -->
    <phase id="8" name="Main Execution Pipeline">
        <task id="8.1" priority="high">
            <name>Create main.py</name>
            <description>Master script to run entire pipeline</description>
            <filepath>main.py</filepath>
            <structure>
                <section name="imports">
                    <import>from src.data_loader import load_data, create_target_column, train_test_split_timeseries</import>
                    <import>from src.feature_engineering import prepare_features</import>
                    <import>from src.models import train_logistic_regression, train_random_forest, train_xgboost, save_model</import>
                    <import>from src.evaluation import evaluate_model, compare_models, plot_confusion_matrix, plot_feature_importance</import>
                    <import>from src.pnl_calculator import generate_signals, calculate_pnl, plot_pnl_curve</import>
                    <import>from sklearn.preprocessing import StandardScaler</import>
                    <import>import joblib</import>
                </section>
                <section name="main_function">
                    <step number="1">
                        <name>Load Data</name>
                        <code>df = load_data('data/raw/nifty_intraday.csv')</code>
                    </step>
                    <step number="2">
                        <name>Create Target</name>
                        <code>df = create_target_column(df)</code>
                    </step>
                    <step number="3">
                        <name>Feature Engineering</name>
                        <code>df = prepare_features(df)</code>
                    </step>
                    <step number="4">
                        <name>Train-Test Split</name>
                        <code>X_train, X_test, y_train, y_test, test_df = train_test_split_timeseries(df)</code>
                    </step>
                    <step number="5">
                        <name>Feature Scaling</name>
                        <code>
                            scaler = StandardScaler()
                            X_train_scaled = scaler.fit_transform(X_train)
                            X_test_scaled = scaler.transform(X_test)
                            joblib.dump(scaler, 'models/scaler.pkl')
                        </code>
                    </step>
                    <step number="6">
                        <name>Train Models</name>
                        <code>
                            lr_model = train_logistic_regression(X_train_scaled, y_train)
                            rf_model = train_random_forest(X_train_scaled, y_train)
                            xgb_model = train_xgboost(X_train_scaled, y_train)
                        </code>
                    </step>
                    <step number="7">
                        <name>Save Models</name>
                        <code>
                            save_model(lr_model, 'logistic_regression')
                            save_model(rf_model, 'random_forest')
                            save_model(xgb_model, 'xgboost')
                        </code>
                    </step>
                    <step number="8">
                        <name>Evaluate and Compare Models</name>
                        <code>
                            models = {
                                'Logistic Regression': lr_model,
                                'Random Forest': rf_model,
                                'XGBoost': xgb_model
                            }
                            best_model_name = compare_models(models, X_test_scaled, y_test)
                            best_model = models[best_model_name]
                        </code>
                    </step>
                    <step number="9">
                        <name>Generate Signals</name>
                        <code>test_df = generate_signals(best_model, test_df, X_test_scaled)</code>
                    </step>
                    <step number="10">
                        <name>Calculate PnL</name>
                        <code>test_df = calculate_pnl(test_df)</code>
                    </step>
                    <step number="11">
                        <name>Plot Results</name>
                        <code>
                            plot_confusion_matrix(y_test, test_df['Predicted'], best_model_name)
                            plot_pnl_curve(test_df)
                        </code>
                    </step>
                    <step number="12">
                        <name>Save Final Output</name>
                        <code>
                            final_output = test_df[['Timestamp', 'Close', 'Predicted', 'model_call', 'model_pnl']]
                            final_output.to_csv('data/processed/final_predictions.csv', index=False)
                        </code>
                    </step>
                    <step number="13">
                        <name>Print Summary</name>
                        <code>
                            print(f"\n{'='*50}")
                            print(f"BEST MODEL: {best_model_name}")
                            print(f"Final PnL: {test_df['model_pnl'].iloc[-1]:.2f}")
                            print(f"Output saved to: data/processed/final_predictions.csv")
                            print(f"{'='*50}")
                        </code>
                    </step>
                </section>
            </structure>
        </task>

        <task id="8.2" priority="high">
            <name>Add execution guard</name>
            <description>Enable script to run directly</description>
            <code>
                if __name__ == "__main__":
                    main()
            </code>
        </task>
    </phase>

    <!-- PHASE 9: DOCUMENTATION -->
    <phase id="9" name="Documentation and README">
        <task id="9.1" priority="high">
            <name>Create comprehensive README.md</name>
            <description>Document project setup and usage</description>
            <filepath>README.md</filepath>
            <sections>
                <section name="Title and Description">
                    <content>
                        # NIFTY Intraday Price Prediction using Machine Learning
                        
                        This project predicts the direction of the next candle's closing price using historical NIFTY intraday data and machine learning models.
                    </content>
                </section>
                <section name="Objective">
                    <content>
                        ## 🎯 Objective
                        - Build ML models to predict if next candle close will be higher (1) or lower (0)
                        - Compare multiple models (Logistic Regression, Random Forest, XGBoost)
                        - Generate trading signals and calculate cumulative PnL
                    </content>
                </section>
                <section name="Dataset">
                    <content>
                        ## 📊 Dataset
                        - **Source**: 1 year of NIFTY intraday OHLC data
                        - **Columns**: Timestamp, Open, High, Low, Close
                        - **Location**: `data/raw/` folder
                    </content>
                </section>
                <section name="Setup Instructions">
                    <content>
                        ## 🚀