#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 13:10:12 2025

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


def bootstrap_evaluation(model, X_test, y_test, model_name, n_iterations=1000):
    """
    Perform bootstrapping to evaluate model performance with confidence intervals.
    
    Parameters:
        model: Trained ML model.
        X_test (np.array): Test features.
        y_test (np.array): True labels.
        model_name (str): Name of the model for display purposes.
        n_iterations (int): Number of bootstrap samples.
    
    Returns:
        dict: Contains mean and confidence intervals for each metric.
    """
    metrics = {"roc_auc": [], "accuracy": [], "precision": [], "recall": [], "f1": []}
    
    for _ in range(n_iterations):
        # Resample test data with replacement
        X_resampled, y_resampled = resample(X_test, y_test, random_state=_)
        
        # Predict probabilities and classes
        y_probs = model.predict_proba(X_resampled)[:, 1] if hasattr(model, "predict_proba") else None
        y_pred = model.predict(X_resampled)
        
        # Calculate metrics
        if y_probs is not None:
            metrics["roc_auc"].append(roc_auc_score(y_resampled, y_probs))
        metrics["accuracy"].append(accuracy_score(y_resampled, y_pred))
        metrics["precision"].append(precision_score(y_resampled, y_pred))
        metrics["recall"].append(recall_score(y_resampled, y_pred))
        metrics["f1"].append(f1_score(y_resampled, y_pred))
    
    # Calculate mean and confidence intervals for each metric
    results = {}
    for metric, values in metrics.items():
        if values:  # Ensure there are values for the metric
            mean = np.mean(values)
            lower = np.percentile(values, 2.5)  # 2.5th percentile for 95% CI
            upper = np.percentile(values, 97.5)  # 97.5th percentile for 95% CI
            results[metric] = (mean, lower, upper)
    
    print(f"\n{model_name} Bootstrap Results:")
    for metric, (mean, lower, upper) in results.items():
        print(f"{metric.upper()}: Mean = {mean:.4f}, 95% CI = [{lower:.4f}, {upper:.4f}]")
    
    return results


def main():
    # Define paths
    embeddings_dir = '/Users/u0145079/Library/CloudStorage/OneDrive-KULeuven/Desktop/Doctorate/PlasticEnz_toolset/Model_making/6_embedings/Mean-pooling'
    models_dir = '/Users/u0145079/Library/CloudStorage/OneDrive-KULeuven/Desktop/Doctorate/PlasticEnz_toolset/Model_making/7_model_selection/Mean-pooling-embs'
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
    
    # Perform bootstrap evaluation for each model
    all_results = []
    for model_name, model_path in model_files.items():
        model = load_model(model_path)
        results = bootstrap_evaluation(model, X_test, y_test, model_name)
        # Collect results for each metric
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
    results_df.to_csv(os.path.join(models_dir, "bootstrap_evaluation_results.csv"), index=False)
    print("\nBootstrap evaluation results saved to 'bootstrap_evaluation_results.csv'.")
    
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
        plt.title(f"Bootstrap Results: {metric}")
        plt.ylabel(f"Mean {metric}")
        plt.xlabel("Model")
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(models_dir, f"{metric.lower()}_bootstrap_plot.png"))
        plt.show()


if __name__ == "__main__":
    main()
