"""
Model evaluation module for performance metrics and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import os


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate model performance and return metrics.
    
    Parameters:
    -----------
    model : sklearn/xgboost model
        Trained model
    X_test : array-like
        Test features
    y_test : array-like
        Test target
    model_name : str
        Name of the model
        
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    print("\n" + "="*60)
    print(f"Evaluating {model_name}...")
    print("="*60)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Print results
    print(f"\n{model_name} Performance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Return metrics as dictionary
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }
    
    return metrics


def compare_models(models_dict, X_test, y_test, output_dir='results'):
    """
    Compare multiple models and select the best one.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary of {model_name: model}
    X_test : array-like
        Test features
    y_test : array-like
        Test target
    output_dir : str
        Directory to save comparison results
        
    Returns:
    --------
    str
        Name of the best model based on accuracy
    """
    print("\n" + "="*60)
    print("COMPARING MODELS")
    print("="*60)
    
    results = []
    
    # Evaluate each model
    for model_name, model in models_dict.items():
        metrics = evaluate_model(model, X_test, y_test, model_name)
        results.append(metrics)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
    
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(comparison_df.to_string(index=False))
    print("="*60)
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    comparison_path = os.path.join(output_dir, 'model_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nComparison saved to: {comparison_path}")
    
    # Get best model name
    best_model_name = comparison_df.iloc[0]['Model']
    best_accuracy = comparison_df.iloc[0]['Accuracy']
    
    print(f"\n🏆 BEST MODEL: {best_model_name} (Accuracy: {best_accuracy:.4f})")
    
    return best_model_name


def plot_confusion_matrix(y_test, y_pred, model_name="Model", output_dir='results'):
    """
    Plot and save confusion matrix.
    
    Parameters:
    -----------
    y_test : array-like
        True labels
    y_pred : array-like
        Predicted labels
    model_name : str
        Name of the model
    output_dir : str
        Directory to save the plot
    """
    print(f"\nGenerating confusion matrix for {model_name}...")
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Create figure
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Sell (0)', 'Buy (1)'],
                yticklabels=['Sell (0)', 'Buy (1)'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f'confusion_matrix_{model_name.replace(" ", "_")}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to: {filepath}")


def plot_feature_importance(model, feature_names, model_name="Model", top_n=20, output_dir='results'):
    """
    Plot and save feature importance for tree-based models.
    
    Parameters:
    -----------
    model : sklearn/xgboost model
        Trained model with feature_importances_ attribute
    feature_names : list
        List of feature names
    model_name : str
        Name of the model
    top_n : int
        Number of top features to display
    output_dir : str
        Directory to save the plot
    """
    # Check if model has feature importances
    if not hasattr(model, 'feature_importances_'):
        print(f"Model {model_name} does not have feature_importances_ attribute. Skipping...")
        return
    
    print(f"\nGenerating feature importance plot for {model_name}...")
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create DataFrame
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Sort by importance
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
    
    # Select top N features
    top_features = feature_importance_df.head(top_n)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Feature Importances - {model_name}')
    plt.gca().invert_yaxis()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f'feature_importance_{model_name.replace(" ", "_")}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Feature importance plot saved to: {filepath}")
    
    # Also save feature importance to CSV
    csv_path = os.path.join(output_dir, f'feature_importance_{model_name.replace(" ", "_")}.csv')
    feature_importance_df.to_csv(csv_path, index=False)
    print(f"Feature importance data saved to: {csv_path}")


def plot_roc_curve(model, X_test, y_test, model_name="Model", output_dir='results'):
    """
    Plot ROC curve for the model.
    
    Parameters:
    -----------
    model : sklearn/xgboost model
        Trained model
    X_test : array-like
        Test features
    y_test : array-like
        Test target
    model_name : str
        Name of the model
    output_dir : str
        Directory to save the plot
    """
    from sklearn.metrics import roc_curve, auc
    
    print(f"\nGenerating ROC curve for {model_name}...")
    
    # Get prediction probabilities
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        print(f"Model {model_name} does not support predict_proba. Skipping ROC curve...")
        return
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Create figure
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f'roc_curve_{model_name.replace(" ", "_")}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ROC curve saved to: {filepath}")
    print(f"AUC Score: {roc_auc:.4f}")
