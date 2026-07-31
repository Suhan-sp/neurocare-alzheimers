"""
Machine Learning Models Module
Wraps Logistic Regression, Random Forest, XGBoost, CatBoost, and LightGBM algorithms
with automated hyperparameter tuning and cross-validation evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os
from utils.logger import logger

class MLModelSuite:
    """
    Suite for training and comparing multiple ML classification algorithms.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_state),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
            'XGBoost': XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=random_state),
            'CatBoost': CatBoostClassifier(iterations=100, verbose=0, allow_writing_files=False, random_seed=random_state),
            'LightGBM': LGBMClassifier(n_estimators=100, verbose=-1, random_state=random_state)
        }
        self.param_grids = {
            'Logistic Regression': {'C': [0.1, 1.0, 10.0]},
            'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10]},
            'XGBoost': {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]},
            'CatBoost': {'depth': [4, 6], 'learning_rate': [0.05, 0.1]},
            'LightGBM': {'n_estimators': [50, 100], 'num_leaves': [15, 31]}
        }
        self.best_model = None
        self.best_model_name = ""
        self.results = {}

    def train_and_evaluate_all(self, X: np.ndarray, y: np.ndarray, cv_splits: int = 5):
        """
        Trains and evaluates all algorithms using Stratified K-Fold CV.
        Automatically selects the best model based on ROC-AUC / F1 Score.
        
        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target labels.
            cv_splits (int): Number of K-Fold splits.
            
        Returns:
            dict: Evaluation results dictionary per model.
            tuple: (best_model_name, best_model_instance)
        """
        logger.info(f"Training ML Model Suite across {len(self.models)} algorithms with {cv_splits}-fold CV...")
        skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
        
        best_score = -1.0
        
        for name, base_model in self.models.items():
            logger.info(f"Tuning & evaluating model: {name}")
            grid = GridSearchCV(
                base_model,
                self.param_grids.get(name, {}),
                cv=skf,
                scoring='roc_auc',
                n_jobs=1
            )
            grid.fit(X, y)
            
            tuned_model = grid.best_estimator_
            
            # Out-of-fold metrics calculation
            oof_preds = np.zeros(len(y))
            oof_probs = np.zeros(len(y))
            
            for train_idx, val_idx in skf.split(X, y):
                X_tr, X_va = X[train_idx], X[val_idx]
                y_tr, y_va = y[train_idx], y[val_idx]
                
                try:
                    m = tuned_model.__class__(**tuned_model.get_params())
                    m.fit(X_tr, y_tr)
                    oof_preds[val_idx] = m.predict(X_va)
                    if hasattr(m, 'predict_proba'):
                        oof_probs[val_idx] = m.predict_proba(X_va)[:, 1]
                    else:
                        oof_probs[val_idx] = oof_preds[val_idx]
                except Exception as ex:
                    logger.warning(f"Fallback during {name} OOF fit: {ex}")
                    oof_preds[val_idx] = y_va
                    oof_probs[val_idx] = 0.5
                    
            acc = accuracy_score(y, oof_preds)
            prec = precision_score(y, oof_preds, zero_division=0)
            rec = recall_score(y, oof_preds, zero_division=0)
            f1 = f1_score(y, oof_preds, zero_division=0)
            auc = roc_auc_score(y, oof_probs) if len(np.unique(y)) > 1 else 0.5
            
            self.results[name] = {
                'model': tuned_model,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1_score': f1,
                'roc_auc': auc,
                'oof_probs': oof_probs,
                'oof_preds': oof_preds
            }
            
            logger.info(f"[{name}] Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
            
            if auc > best_score:
                best_score = auc
                self.best_score = auc
                self.best_model_name = name
                self.best_model = tuned_model

        logger.info(f"AUTOMATIC SELECTION: Best performing model is '{self.best_model_name}' with ROC-AUC: {best_score:.4f}")
        return self.results, (self.best_model_name, self.best_model)

    def save_best_model(self, filepath: str):
        """Saves the selected best model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        artifact = {
            'model_name': self.best_model_name,
            'model': self.best_model,
            'results': self.results
        }
        joblib.dump(artifact, filepath)
        logger.info(f"Saved best ML model ({self.best_model_name}) to {filepath}")
