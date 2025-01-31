#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 15:58:48 2024

@author: u0145079

Evaluate trained models on the testing dataset,
collect metrics, and compare.
"""

import os
import numpy as np
import pickle
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    average_precision_score
)

def load_data(embeddings_dir):
    """
    Load test embeddings and labels.
    
    Parameters:
        embeddings_dir (str): Path to the embeddings directory.
        
    Returns:
        X_test, y_test
    """
    X_test = np.load(os.path.join(embeddings_dir, 'test_embeddings.npy'))
    y_test = np.load(os.path.join(embeddings_dir, 'test_labels.npy'))
    return X_test, y_test

def load_model(model_path):
    """
    Load a trained model from a pickle file.
    
    Parameters:
        model_path (str): Path to the model file.
        
    Returns:
        model
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate the model and collect metrics.
    
    Parameters:
        model: Trained ML model.
        X_test (np.array): Test features.
        y_test (np.array): True labels.
        model_name (str): Name of the model for display purposes.
    
    Returns:
        dict: Contains evaluation metrics.
    """
    y_probs = model.predict_proba(X_test)[:,1]
    y_pred = (y_probs >= 0.5).astype(int)
    
    roc_auc = roc_auc_score(y_test, y_probs)
    avg_precision = average_precision_score(y_test, y_probs)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        'Model': model_name,
        'ROC_AUC': roc_auc,
        'Average_Precision': avg_precision,
        'Precision': report['1']['precision'],
        'Recall': report['1']['recall'],
        'F1_Score': report['1']['f1-score'],
        'Support': report['1']['support'],
        'Confusion_Matrix': cm
    }
    
    return metrics

def plot_metrics(metrics_df):
    """
    Plot comparison of ROC AUC and Average Precision across models.
    """
    plt.figure(figsize=(10,6))
    sns.barplot(x='Model', y='ROC_AUC', data=metrics_df, palette='viridis')
    plt.ylim(0, 1)
    plt.title('ROC AUC Scores of Models')
    plt.ylabel('ROC AUC Score')
    plt.xlabel('Model')
    plt.show()
    
    plt.figure(figsize=(10,6))
    sns.barplot(x='Model', y='Average_Precision', data=metrics_df, palette='magma')
    plt.ylim(0, 1)
    plt.title('Average Precision (PR AUC) of Models')
    plt.ylabel('Average Precision (PR AUC)')
    plt.xlabel('Model')
    plt.show()
    
    plt.figure(figsize=(10,6))
    sns.barplot(x='Model', y='F1_Score', data=metrics_df, palette='coolwarm')
    plt.ylim(0, 1)
    plt.title('F1 Scores of Models')
    plt.ylabel('F1 Score')
    plt.xlabel('Model')
    plt.show()

def main():
    # Define paths
    data_dir = '/Users/u0145079/Library/CloudStorage/OneDrive-KULeuven/Desktop/Doctorate/PlasticEnz_toolset/Model_making/Model/embeddings'
    models_dir = '/Users/u0145079/Library/CloudStorage/OneDrive-KULeuven/Desktop/Doctorate/PlasticEnz_toolset/Model_making/Model/models'
    
    # Load test data
    print("Loading test embeddings and labels...")
    X_test, y_test = load_data(data_dir)
    print(f"Testing data: {X_test.shape[0]} samples")
    
    # List of models
    models = {
        'Random Forest': os.path.join(models_dir, 'random_forest.pkl'),
        'XGBoost': os.path.join(models_dir, 'xgboost.pkl'),
        'MLP Neural Network': os.path.join(models_dir, 'mlp.pkl')
    }
    
    # Initialize list to store metrics
    metrics_list = []
    
    # Iterate through models
    for model_name, model_path in models.items():
        print(f"\nEvaluating model: {model_name}")
        model = load_model(model_path)
        
        metrics = evaluate_model(model, X_test, y_test, model_name)
        metrics_list.append(metrics)
        
        # Display confusion matrix
        cm = metrics['Confusion_Matrix']
        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Non-Plastic', 'Plastic'],
                    yticklabels=['Non-Plastic', 'Plastic'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.show()
    
    # Create DataFrame of metrics
    metrics_df = pd.DataFrame(metrics_list)
    
    # Display metrics
    print("\nModel Evaluation Metrics:")
    print(metrics_df[['Model', 'ROC_AUC', 'Average_Precision', 'Precision', 'Recall', 'F1_Score']])
    
    # Plot comparison
    plot_metrics(metrics_df)
    
    # Save metrics to CSV
    metrics_df.to_csv(os.path.join(data_dir, 'model_comparison_metrics.csv'), index=False)
    print("Model comparison metrics saved to model_comparison_metrics.csv")
    
    # Determine the best model based on ROC_AUC
    best_model = metrics_df.loc[metrics_df['ROC_AUC'].idxmax()]
    print(f"\nBest Model: {best_model['Model']} with ROC AUC: {best_model['ROC_AUC']:.4f}")

if __name__ == "__main__":
    main()
    

