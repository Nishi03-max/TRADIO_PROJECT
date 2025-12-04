"""
Machine learning models module for NIFTY price prediction.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import os


def train_logistic_regression(X_train, y_train):
    """
    Train Logistic Regression model.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
        
    Returns:
    --------
    LogisticRegression
        Trained model
    """
    print("\n" + "="*60)
    print("Training Logistic Regression...")
    print("="*60)
    
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    print("Logistic Regression training complete!")
    
    return model


def train_random_forest(X_train, y_train):
    """
    Train Random Forest Classifier.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
        
    Returns:
    --------
    RandomForestClassifier
        Trained model
    """
    print("\n" + "="*60)
    print("Training Random Forest...")
    print("="*60)
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    model.fit(X_train, y_train)
    
    print("Random Forest training complete!")
    
    return model


def train_xgboost(X_train, y_train):
    """
    Train XGBoost Classifier.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
        
    Returns:
    --------
    XGBClassifier
        Trained model
    """
    print("\n" + "="*60)
    print("Training XGBoost...")
    print("="*60)
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1,
        verbosity=0
    )
    
    model.fit(X_train, y_train)
    
    print("XGBoost training complete!")
    
    return model


def save_model(model, filename, output_dir='models'):
    """
    Save trained model to disk.
    
    Parameters:
    -----------
    model : sklearn/xgboost model
        Trained model
    filename : str
        Name of the file (without extension)
    output_dir : str
        Directory to save the model
    """
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, f'{filename}.pkl')
    joblib.dump(model, filepath)
    
    print(f"Model saved to: {filepath}")


def load_model(filename, models_dir='models'):
    """
    Load trained model from disk.
    
    Parameters:
    -----------
    filename : str
        Name of the file (without extension)
    models_dir : str
        Directory containing the model
        
    Returns:
    --------
    model
        Loaded model
    """
    filepath = os.path.join(models_dir, f'{filename}.pkl')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    model = joblib.load(filepath)
    print(f"Model loaded from: {filepath}")
    
    return model


def train_all_models(X_train, y_train):
    """
    Train all models and return as dictionary.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
        
    Returns:
    --------
    dict
        Dictionary of trained models
    """
    models = {}
    
    # Train Logistic Regression
    models['Logistic Regression'] = train_logistic_regression(X_train, y_train)
    
    # Train Random Forest
    models['Random Forest'] = train_random_forest(X_train, y_train)
    
    # Train XGBoost
    models['XGBoost'] = train_xgboost(X_train, y_train)
    
    return models
