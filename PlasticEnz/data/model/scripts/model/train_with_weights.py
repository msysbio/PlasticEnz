#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 17:56:45 2025

@author: u0145079
"""

import os
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from collections import Counter

def load_training_data(embeddings_dir):
    """
    Load training embeddings and labels.
    """
    X_train = np.load(os.path.join(embeddings_dir, 'train_embeddings.npy'))
    y_train = np.load(os.path.join(embeddings_dir, 'train_labels.npy'))
    return X_train, y_train

def compute_class_weights(y_train):
    """
    Compute class weights based on training data.
    """
    counter = Counter(y_train)
    total_samples = len(y_train)
    class_weights = {cls: total_samples / count for cls, count in counter.items()}
    return class_weights

def train_random_forest(X_train, y_train, class_weights):
    """
    Train and tune a Random Forest classifier with class weights.
    """
    rf = RandomForestClassifier(random_state=42, class_weight=class_weights)
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X_train, y_train)
    
    print("Best Random Forest parameters:", grid_search.best_params_)
    print("Best Random Forest ROC AUC:", grid_search.best_score_)
    
    best_rf = grid_search.best_estimator_
    return best_rf

def train_xgboost(X_train, y_train, scale_pos_weight):
    """
    Train and tune an XGBoost classifier with scale_pos_weight.
    """
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=scale_pos_weight)
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X_train, y_train)
    
    print("Best XGBoost parameters:", grid_search.best_params_)
    print("Best XGBoost ROC AUC:", grid_search.best_score_)
    
    best_xgb = grid_search.best_estimator_
    return best_xgb

def train_mlp(X_train, y_train, class_weights):
    """
    Train and tune a shallow Neural Network (MLP) with manual handling of class weights.
    
    Parameters:
        X_train (np.array): Training embeddings.
        y_train (np.array): Training labels.
        class_weights (dict): Class weights to address imbalance.
        
    Returns:
        best_mlp: Best MLP model after tuning.
    """
    # Define the MLP model
    mlp = MLPClassifier(random_state=42, max_iter=500)
    
    # Calculate sample weights based on the provided class weights
    sample_weights = np.array([class_weights[label] for label in y_train])
    
    # Define the parameter grid
    param_grid = {
        'hidden_layer_sizes': [(128, 64), (256, 128), (64, 32)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive']
    }
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=mlp,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=2
    )
    
    # Fit the grid search with sample weights
    grid_search.fit(X_train, y_train, sample_weight=sample_weights)
    
    print("Best MLP parameters:", grid_search.best_params_)
    print("Best MLP ROC AUC:", grid_search.best_score_)
    
    # Return the best MLP model
    best_mlp = grid_search.best_estimator_
    return best_mlp

def main():
    # Define paths
    embeddings_dir = '/Users/u0145079/Library/CloudStorage/OneDrive-KULeuven/Desktop/Doctorate/PlasticEnz_toolset/Model_making/6_embedings/Mean-pooling'
    models_dir = '/Users/u0145079/Library/CloudStorage/OneDrive-KULeuven/Desktop/Doctorate/PlasticEnz_toolset/Model_making/7_model_selection/Mean-pooling-embs/trained_models_with_weight'
    os.makedirs(models_dir, exist_ok=True)
    
    # Load training data
    print("Loading training data...")
    X_train, y_train = load_training_data(embeddings_dir)
    print(f"Training data: {X_train.shape[0]} samples")
    
    # Compute class weights
    class_weights = compute_class_weights(y_train)
    print("Class weights:", class_weights)
    
    # Calculate scale_pos_weight for XGBoost
    scale_pos_weight = class_weights[0] / class_weights[1]
    print("Scale_pos_weight for XGBoost:", scale_pos_weight)
    
    # Train and save models
    print("\nTraining and tuning Random Forest...")
    best_rf = train_random_forest(X_train, y_train, class_weights)
    with open(os.path.join(models_dir, 'random_forest.pkl'), 'wb') as f:
        pickle.dump(best_rf, f)
    print("Random Forest model saved.")
    
    print("\nTraining and tuning XGBoost...")
    best_xgb = train_xgboost(X_train, y_train, scale_pos_weight)
    with open(os.path.join(models_dir, 'xgboost.pkl'), 'wb') as f:
        pickle.dump(best_xgb, f)
    print("XGBoost model saved.")
    
    print("\nTraining and tuning MLP Neural Network...")
    best_mlp = train_mlp(X_train, y_train, class_weights)
    with open(os.path.join(models_dir, 'mlp.pkl'), 'wb') as f:
        pickle.dump(best_mlp, f)
    print("MLP Neural Network model saved.")

if __name__ == "__main__":
    main()
