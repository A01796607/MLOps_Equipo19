"""
Model Trainer for training and optimizing ML models.
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

# Optional dependencies
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None  # type: ignore

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None  # type: ignore


class ModelTrainer:
    """Class for training and optimizing ML models."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize ModelTrainer.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.models: Dict[str, any] = {}
        self.predictions: Dict[str, np.ndarray] = {}
    
    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_estimators: int = 200
    ) -> RandomForestClassifier:
        """
        Train Random Forest classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_estimators: Number of trees
            
        Returns:
            Trained Random Forest model
        """
        logger.info("Training Random Forest classifier...")
        
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=self.random_state
        )
        
        clf.fit(X_train, y_train)
        
        self.models['random_forest'] = clf
        logger.success("Random Forest training complete")
        
        return clf
    
    def train_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        n_estimators: int = 500,
        early_stopping_rounds: int = 5
    ):
        """
        Train LightGBM model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            n_estimators: Number of boosting rounds
            early_stopping_rounds: Early stopping rounds
            
        Returns:
            Trained LightGBM model
            
        Raises:
            ImportError: If lightgbm is not installed
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed. Install with: pip install lightgbm")
        
        logger.info("Training LightGBM model...")
        
        params = {
            'objective': 'multiclass',
            'num_class': len(np.unique(y_train)),
            'boosting_type': 'gbdt',
            'metric': 'multi_logloss',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': -1,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': self.random_state
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets = [train_data, val_data]
        else:
            valid_sets = [train_data]
        
        callbacks = []
        if early_stopping_rounds > 0 and X_val is not None:
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds))
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=valid_sets,
            num_boost_round=n_estimators,
            callbacks=callbacks
        )
        
        self.models['lightgbm'] = model
        logger.success("LightGBM training complete")
        
        return model
    
    def optimize_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_trials: int = 50,
        n_folds: int = 5
    ) -> Tuple:
        """
        Optimize LightGBM hyperparameters using Optuna.
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_trials: Number of optimization trials
            n_folds: Number of CV folds
            
        Returns:
            Tuple of (best model, best parameters)
            
        Raises:
            ImportError: If lightgbm or optuna is not installed
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed. Install with: pip install lightgbm")
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is not installed. Install with: pip install optuna")
        
        logger.info(f"Optimizing LightGBM with {n_trials} trials...")
        
        def objective(trial):
            params = {
                'objective': 'multiclass',
                'num_class': len(np.unique(y_train)),
                'boosting_type': 'gbdt',
                'metric': 'multi_logloss',
                'verbosity': -1,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 5.0),
                'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 5.0)
            }
            
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
            f1_scores = []
            
            for train_idx, val_idx in kf.split(X_train, y_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                
                model = lgb.LGBMClassifier(**params, n_estimators=500, verbose=-1)
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    eval_metric='multi_logloss',
                    callbacks=[lgb.early_stopping(stopping_rounds=5)]
                )
                
                preds = model.predict(X_val)
                f1 = f1_score(y_val, preds, average='macro')
                f1_scores.append(f1)
            
            return np.mean(f1_scores)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        logger.info(f"Best trial: {study.best_trial.params}")
        
        best_params = study.best_trial.params.copy()
        best_params.update({
            'objective': 'multiclass',
            'num_class': len(np.unique(y_train)),
            'boosting_type': 'gbdt',
            'metric': 'multi_logloss'
        })
        
        # Train final model with best parameters
        final_model = lgb.LGBMClassifier(**best_params, n_estimators=800)
        final_model.fit(X_train, y_train)
        
        self.models['lightgbm_optimized'] = final_model
        logger.success("LightGBM optimization complete")
        
        return final_model, best_params
    
    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using a trained model.
        
        Args:
            model_name: Name of the model to use
            X: Features for prediction
            
        Returns:
            Predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        
        # Check if it's a LightGBM Booster (if lightgbm is available)
        if LIGHTGBM_AVAILABLE and lgb is not None:
            try:
                # Try to check if it's a Booster
                if hasattr(model, 'predict') and hasattr(model, 'num_trees'):
                    # Likely a LightGBM Booster
                    y_pred_proba = model.predict(X)
                    y_pred = y_pred_proba.argmax(axis=1)
                else:
                    # Regular sklearn-style model
                    y_pred = model.predict(X)
            except Exception:
                # Fallback to regular predict
                y_pred = model.predict(X)
        else:
            # Regular sklearn-style model
            y_pred = model.predict(X)
        
        self.predictions[model_name] = y_pred
        return y_pred
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[list] = None
    ) -> dict:
        """
        Evaluate model predictions.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            
        Returns:
            Dictionary with evaluation metrics
        """
        report = classification_report(
            y_true, y_pred,
            target_names=class_names,
            output_dict=True
        )
        
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'classification_report': report,
            'confusion_matrix': cm
        }
    
    def print_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[list] = None,
        model_name: str = ""
    ):
        """
        Print classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            model_name: Name of the model
        """
        print(f"\n{'='*50}")
        print(f"Classification Report {model_name}")
        print(f"{'='*50}")
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))

