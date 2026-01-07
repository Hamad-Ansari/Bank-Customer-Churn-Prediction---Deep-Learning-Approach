"""
Model training module for bank churn prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import joblib
import yaml
import os
from datetime import datetime


class ModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self, config_path='config/params.yaml'):
        """
        Initialize model trainer with configuration
        
        Args:
            config_path (str): Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.models = {}
        self.results = {}
        
    def train_lightgbm(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train LightGBM model
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training target
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation target
            
        Returns:
            Trained LightGBM model
        """
        params = self.config['lightgbm']
        
        if X_val is not None and y_val is not None:
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
        else:
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train)
        
        self.models['lightgbm'] = model
        return model
    
    def train_xgboost(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train XGBoost model
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training target
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation target
            
        Returns:
            Trained XGBoost model
        """
        params = self.config['xgboost']
        
        if X_val is not None and y_val is not None:
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False
            )
        else:
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train)
        
        self.models['xgboost'] = model
        return model
    
    def train_catboost(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train CatBoost model
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training target
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation target
            
        Returns:
            Trained CatBoost model
        """
        params = self.config['catboost']
        
        if X_val is not None and y_val is not None:
            model = CatBoostClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=50,
                verbose=False
            )
        else:
            model = CatBoostClassifier(**params)
            model.fit(X_train, y_train, verbose=False)
        
        self.models['catboost'] = model
        return model
    
    def evaluate_model(self, model, X_val, y_val, model_name='model'):
        """
        Evaluate model performance
        
        Args:
            model: Trained model
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation target
            model_name (str): Name of the model
            
        Returns:
            dict: Evaluation metrics
        """
        # Predictions
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0),
            'f1': f1_score(y_val, y_pred, zero_division=0)
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_val, y_pred_proba)
        
        # Store results
        self.results[model_name] = metrics
        
        # Print results
        print(f"\n{model_name.upper()} Performance:")
        print("-" * 40)
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        
        return metrics
    
    def cross_validate(self, model, X, y, n_splits=5):
        """
        Perform cross-validation
        
        Args:
            model: Model to cross-validate
            X (np.ndarray): Features
            y (np.ndarray): Target
            n_splits (int): Number of cross-validation folds
            
        Returns:
            dict: Cross-validation results
        """
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, 
                           random_state=self.config['model']['random_state'])
        
        cv_scores = cross_val_score(
            model, X, y,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        return {
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'scores': cv_scores
        }
    
    def save_model(self, model, model_name, path='models/'):
        """
        Save trained model
        
        Args:
            model: Trained model to save
            model_name (str): Name of the model
            path (str): Directory to save the model
        """
        os.makedirs(path, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{path}{model_name}_{timestamp}.pkl"
        
        joblib.dump(model, filename)
        print(f"Model saved to {filename}")
        
        return filename
    
    def save_results(self, path='reports/model_results.csv'):
        """
        Save model results to CSV
        
        Args:
            path (str): Path to save results
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        results_df = pd.DataFrame(self.results).T
        results_df.to_csv(path)
        print(f"Results saved to {path}")
        
        return results_df


def main():
    """Main function for testing the module"""
    # Load preprocessed data
    preprocessor = DataPreprocessor()
    train_df, test_df = preprocessor.load_data()
    processed_data = preprocessor.preprocess_data(train_df, test_df)
    
    # Train models
    trainer = ModelTrainer()
    
    # Train and evaluate each model
    for model_name in ['lightgbm', 'xgboost', 'catboost']:
        print(f"\nTraining {model_name.upper()}...")
        model_func = getattr(trainer, f'train_{model_name}')
        model = model_func(
            processed_data['X_train'],
            processed_data['y_train'],
            processed_data['X_val'],
            processed_data['y_val']
        )
        
        # Evaluate
        trainer.evaluate_model(
            model,
            processed_data['X_val'],
            processed_data['y_val'],
            model_name
        )
        
        # Save model
        trainer.save_model(model, model_name)
    
    # Save results
    results_df = trainer.save_results()
    
    return trainer, results_df


if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor
    trainer, results = main()
