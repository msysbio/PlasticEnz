#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 14:43:11 2024

@author: u0145079

Train and tune Random Forest, XGBoost, and MLP classifiers.
Hyperparameter tuning is performed using GridSearchCV.
"""

import os
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.metrics import roc_auc_score

def load_data(embeddings_dir):
    """
    Load embeddings and labels from the specified directory.
    
    Parameters:
        embeddings_dir (str): Path to the embeddings directory.
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train = np.load(os.path.join(embeddings_dir, 'train_embeddings.npy'))
    X_test = np.load(os.path.join(embeddings_dir, 'test_embeddings.npy'))
    y_train = np.load(os.path.join(embeddings_dir, 'train_labels.npy'))
    y_test = np.load(os.path.join(embeddings_dir, 'test_labels.npy'))
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, y_train):
    """
    Train and tune a Random Forest classifier using GridSearchCV.
    
    Parameters:
        X_train (np.array): Training embeddings.
        y_train (np.array): Training labels.
        
    Returns:
        best_rf: Best Random Forest model after tuning.
    """
    rf = RandomForestClassifier(random_state=42)
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
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

def train_xgboost(X_train, y_train):
    """
    Train and tune an XGBoost classifier using GridSearchCV.
    
    Parameters:
        X_train (np.array): Training embeddings.
        y_train (np.array): Training labels.
        
    Returns:
        best_xgb: Best XGBoost model after tuning.
    """
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    
    param_grid = {
        'n_estimators': [100, 200],
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

def train_mlp(X_train, y_train):
    """
    Train and tune a shallow Neural Network (MLP) using GridSearchCV.
    
    Parameters:
        X_train (np.array): Training embeddings.
        y_train (np.array): Training labels.
        
    Returns:
        best_mlp: Best MLP model after tuning.
    """
    mlp = MLPClassifier(random_state=42, max_iter=500)
    
    param_grid = {
        'hidden_layer_sizes': [(128, 64), (256, 128), (64, 32)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001],
        'learning_rate': ['constant', 'adaptive']
    }
    
    grid_search = GridSearchCV(
        estimator=mlp,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X_train, y_train)
    
    print("Best MLP parameters:", grid_search.best_params_)
    print("Best MLP ROC AUC:", grid_search.best_score_)
    
    best_mlp = grid_search.best_estimator_
    
    return best_mlp

def main():
    # Define paths
    embeddings_dir = '/Users/u0145079/Library/CloudStorage/OneDrive-KULeuven/Desktop/Doctorate/PlasticEnz_toolset/Model_making/Model/embeddings'
    models_dir = '/Users/u0145079/Library/CloudStorage/OneDrive-KULeuven/Desktop/Doctorate/PlasticEnz_toolset/Model_making/Model/models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Load data
    print("Loading embeddings and labels...")
    X_train, X_test, y_train, y_test = load_data(embeddings_dir)
    print(f"Training data: {X_train.shape[0]} samples")
    print(f"Testing data: {X_test.shape[0]} samples")
    
    # Train and tune Random Forest
    print("\nTraining and tuning Random Forest...")
    best_rf = train_random_forest(X_train, y_train)
    # Save the model
    with open(os.path.join(models_dir, 'random_forest.pkl'), 'wb') as f:
        pickle.dump(best_rf, f)
    print("Random Forest model saved.")
    
    # Train and tune XGBoost
    print("\nTraining and tuning XGBoost...")
    best_xgb = train_xgboost(X_train, y_train)
    # Save the model
    with open(os.path.join(models_dir, 'xgboost.pkl'), 'wb') as f:
        pickle.dump(best_xgb, f)
    print("XGBoost model saved.")
    
    # Train and tune MLP
    print("\nTraining and tuning MLP Neural Network...")
    best_mlp = train_mlp(X_train, y_train)
    # Save the model
    with open(os.path.join(models_dir, 'mlp.pkl'), 'wb') as f:
        pickle.dump(best_mlp, f)
    print("MLP Neural Network model saved.")

if __name__ == "__main__":
    main()
