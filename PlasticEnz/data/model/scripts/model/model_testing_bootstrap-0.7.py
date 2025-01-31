#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 10:15:07 2025

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
from sklearn.utils import resample


def load_testing_data(embeddings_dir):
    """
    Load testing embeddings and labels.
    """
    X_test = np.load(os.path.join(embeddings_dir, 'test_embeddings.npy'))
    y_test = np.load(os.path.join(embeddings_dir, 'test_labels.npy'))
    return X_test, y_test


def load_model(model_path):
    """
    Load a trained model from a pickle file.
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def evaluate_with_threshold(model, X, y, threshold=0.7):
    """
    Evaluate the model using a custom threshold for classification.
    """
    if hasattr(model, "predict_proba"):
        y_probs = model.predict_proba(X)[:, 1]
        y_pred = (y_probs >= threshold).astype(int)
    else:
        # For SVM or models without predict_proba, use standard predictions
        y_probs = None
        y_pred = model.predict(X)
    
    metrics = {
        "roc_auc": roc_auc_score(y, y_probs) if y_probs is not None else None,
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred)
    }
    
    return metrics


def bootstrap_evaluation_with_threshold(model, X_test, y_test, model_name, threshold=0.7, n_iterations=1000):
    """
    Perform bootstrapping to evaluate model performance with a custom threshold.
    """
    metrics = {"roc_auc": [], "accuracy": [], "precision": [], "recall": [], "f1": []}
    
    for _ in range(n_iterations):
        # Resample test data with replacement
        X_resampled, y_resampled = resample(X_test, y_test, random_state=_)
        
        # Evaluate model on resampled data
        resampled_metrics = evaluate_with_threshold(model, X_resampled, y_resampled, threshold)
        for metric_name, value in resampled_metrics.items():
            if value is not None:
                metrics[metric_name].append(value)
    
    # Calculate mean and confidence intervals for each metric
    results = {}
    for metric, values in metrics.items():
        if values:
            mean = np.mean(values)
            lower = np.percentile(values, 2.5)
            upper = np.percentile(values, 97.5)
            results[metric] = (mean, lower, upper)
    
    print(f"\n{model_name} Bootstrap Results (Threshold = {threshold}):")
    for metric, (mean, lower, upper) in results.items():
        print(f"{metric.upper()}: Mean = {mean:.4f}, 95% CI = [{lower:.4f}, {upper:.4f}]")
    
    return results


def main():
    # Define paths
    embeddings_dir = '/Users/u0145079/Library/CloudStorage/OneDrive-KULeuven/Desktop/Doctorate/PlasticEnz_toolset/Current_version/PlasticEnz/PlasticEnz/data/model/embeddings'
    models_dir = '/Users/u0145079/Library/CloudStorage/OneDrive-KULeuven/Desktop/Doctorate/PlasticEnz_toolset/Current_version/PlasticEnz/PlasticEnz/data/model/models'
    model_files = {
        "Random Forest": os.path.join(models_dir, 'random_forest.pkl'),
        "XGBoost": os.path.join(models_dir, 'xgboost.pkl'),
        "MLP": os.path.join(models_dir, 'mlp.pkl'),
        "Linear SVM": os.path.join(models_dir, 'svm.pkl')
    }
    
    # Load testing data
    print("Loading testing data...")
    X_test, y_test = load_testing_data(embeddings_dir)
    print(f"Testing data: {X_test.shape[0]} samples")
    
    # Perform bootstrap evaluation with a custom threshold for each model
    threshold = 0.7
    all_results = []
    for model_name, model_path in model_files.items():
        model = load_model(model_path)
        results = bootstrap_evaluation_with_threshold(model, X_test, y_test, model_name, threshold)
        for metric, (mean, lower, upper) in results.items():
            all_results.append({
                "Model": model_name,
                "Metric": metric.upper(),
                "Mean": mean,
                "95% CI Lower": lower,
                "95% CI Upper": upper
            })
    
    # Save results to a CSV file
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(models_dir, "bootstrap_evaluation_with_threshold_results.csv"), index=False)
    print("\nBootstrap evaluation results saved to 'bootstrap_evaluation_with_threshold_results.csv'.")
    
    # Plot results
    print("\nGenerating bootstrap results plots...")
    sns.set(style="whitegrid")
    for metric in results_df["Metric"].unique():
        metric_df = results_df[results_df["Metric"] == metric]
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Model", y="Mean", data=metric_df, palette="viridis", ci=None)
        for index, row in metric_df.iterrows():
            plt.errorbar(
                x=row["Model"],
                y=row["Mean"],
                yerr=[[row["Mean"] - row["95% CI Lower"]], [row["95% CI Upper"] - row["Mean"]]],
                fmt='none',
                c='black',
                capsize=5
            )
        plt.title(f"Bootstrap Results (Threshold {threshold}): {metric}")
        plt.ylabel(f"Mean {metric}")
        plt.xlabel("Model")
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(models_dir, f"{metric.lower()}_bootstrap_plot_threshold_{threshold}.png"))
        plt.show()


if __name__ == "__main__":
    main()
