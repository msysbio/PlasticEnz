#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 17:01:46 2025

@author: u0145079
"""

import os
import numpy as np
import pickle
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_testing_data(embeddings_dir):
    """
    Load testing embeddings and labels.
    
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
    Evaluate the model on the test set and calculate performance metrics.
    
    Parameters:
        model: Trained ML model.
        X_test (np.array): Test features.
        y_test (np.array): True labels.
        model_name (str): Name of the model for display purposes.
    
    Returns:
        dict: Contains evaluation metrics.
    """
    y_probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    y_pred = model.predict(X_test)
    
    roc_auc = roc_auc_score(y_test, y_probs) if y_probs is not None else None
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        "Model": model_name,
        "ROC AUC": roc_auc,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Confusion Matrix": cm
    }


def plot_metrics(results_df, output_dir):
    """
    Plot evaluation metrics for comparison.
    
    Parameters:
        results_df (pd.DataFrame): DataFrame containing evaluation metrics for all models.
        output_dir (str): Directory to save the plots.
    """
    metrics = ["ROC AUC", "Accuracy", "Precision", "Recall", "F1 Score"]
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Model", y=metric, data=results_df, palette="viridis")
        plt.title(f"Model Comparison: {metric}")
        plt.ylabel(metric)
        plt.xlabel("Model")
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric.lower().replace(' ', '_')}_comparison.png"))
        plt.show()


def main():
    # Define paths
    embeddings_dir = '/Users/u0145079/Library/CloudStorage/OneDrive-KULeuven/Desktop/Doctorate/PlasticEnz_toolset/Model_making/6_embedings/Mean-pooling'
    models_dir = '/Users/u0145079/Library/CloudStorage/OneDrive-KULeuven/Desktop/Doctorate/PlasticEnz_toolset/Model_making/7_model_selection/Mean-pooling-embs/trained_models_with_weight'
    output_dir = os.path.join(models_dir, "evaluation_plots")
    os.makedirs(output_dir, exist_ok=True)
    
    # Include only Random Forest and XGBoost
    model_files = {
        "Random Forest": os.path.join(models_dir, 'random_forest.pkl'),
        "XGBoost": os.path.join(models_dir, 'xgboost.pkl')
    }
    
    # Load testing data
    print("Loading testing data...")
    X_test, y_test = load_testing_data(embeddings_dir)
    print(f"Testing data: {X_test.shape[0]} samples")
    
    # Evaluate each model
    results = []
    for model_name, model_path in model_files.items():
        model = load_model(model_path)
        metrics = evaluate_model(model, X_test, y_test, model_name)
        results.append(metrics)
    
    # Save results to a CSV file
    results_df = pd.DataFrame(results).drop(columns=["Confusion Matrix"])  # Save all metrics except confusion matrix
    results_df.to_csv(os.path.join(models_dir, "model_evaluation_results_rf_xgb.csv"), index=False)
    print("\nModel evaluation results saved to 'model_evaluation_results_rf_xgb.csv'.")
    
    # Plot metrics for comparison
    print("\nGenerating comparison plots...")
    plot_metrics(results_df, output_dir)
    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
