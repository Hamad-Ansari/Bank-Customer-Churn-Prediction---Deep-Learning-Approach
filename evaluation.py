"""
Model evaluation and visualization module
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_curve, auc, confusion_matrix, 
                           precision_recall_curve, classification_report)
import joblib
import yaml
import os


class ModelEvaluator:
    """Handles model evaluation and visualization"""
    
    def __init__(self, config_path='config/params.yaml'):
        """
        Initialize evaluator with configuration
        
        Args:
            config_path (str): Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def plot_roc_curve(self, model, X_test, y_test, model_name='Model'):
        """
        Plot ROC curve
        
        Args:
            model: Trained model
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test target
            model_name (str): Name of the model
        """
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = model.decision_function(X_test)
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Save figure
        os.makedirs('reports/figures', exist_ok=True)
        plt.savefig(f'reports/figures/roc_curve_{model_name.lower()}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return roc_auc
    
    def plot_confusion_matrix(self, model, X_test, y_test, model_name='Model'):
        """
        Plot confusion matrix
        
        Args:
            model: Trained model
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test target
            model_name (str): Name of the model
        """
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Not Churned', 'Churned'],
                   yticklabels=['Not Churned', 'Churned'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title(f'Confusion Matrix - {model_name}')
        
        # Save figure
        os.makedirs('reports/figures', exist_ok=True)
        plt.savefig(f'reports/figures/confusion_matrix_{model_name.lower()}.png',
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
    
    def plot_feature_importance(self, model, feature_names, model_name='Model', top_n=20):
        """
        Plot feature importance
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names (list): List of feature names
            model_name (str): Name of the model
            top_n (int): Number of top features to show
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            print(f"Model {model_name} doesn't have feature importance attribute")
            return None
        
        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        # Sort by importance
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        # Plot top N features
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(top_n)
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        
        bars = plt.barh(range(len(top_features)), top_features['importance'], color=colors)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importance - {model_name}')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, v in enumerate(top_features['importance']):
            plt.text(v + 0.001, i, f'{v:.4f}', va='center')
        
        plt.tight_layout()
        
        # Save figure
        os.makedirs('reports/figures', exist_ok=True)
        plt.savefig(f'reports/figures/feature_importance_{model_name.lower()}.png',
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return feature_importance
    
    def plot_precision_recall_curve(self, model, X_test, y_test, model_name='Model'):
        """
        Plot precision-recall curve
        
        Args:
            model: Trained model
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test target
            model_name (str): Name of the model
        """
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = model.decision_function(X_test)
        
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = np.mean(precision)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='darkgreen', lw=2,
                label=f'Avg Precision = {avg_precision:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        # Save figure
        os.makedirs('reports/figures', exist_ok=True)
        plt.savefig(f'reports/figures/precision_recall_{model_name.lower()}.png',
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return avg_precision
    
    def generate_classification_report(self, model, X_test, y_test, model_name='Model'):
        """
        Generate and save classification report
        
        Args:
            model: Trained model
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test target
            model_name (str): Name of the model
            
        Returns:
            dict: Classification report
        """
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Convert to dataframe
        report_df = pd.DataFrame(report).transpose()
        
        # Save to CSV
        os.makedirs('reports', exist_ok=True)
        report_df.to_csv(f'reports/classification_report_{model_name.lower()}.csv')
        
        # Print report
        print(f"\nClassification Report - {model_name}:")
        print("-" * 60)
        print(classification_report(y_test, y_pred))
        
        return report_df
    
    def compare_models(self, models_dict, X_test, y_test):
        """
        Compare multiple models
        
        Args:
            models_dict (dict): Dictionary of model names and models
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test target
        """
        comparison_results = {}
        
        for model_name, model in models_dict.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            results = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0)
            }
            
            if y_pred_proba is not None:
                results['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
            
            comparison_results[model_name] = results
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(comparison_results).T
        
        # Plot comparison
        plt.figure(figsize=(12, 8))
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        metrics_to_plot = [m for m in metrics_to_plot if m in comparison_df.columns]
        
        comparison_df[metrics_to_plot].plot(kind='bar', figsize=(12, 8))
        plt.title('Model Comparison')
        plt.ylabel('Score')
        plt.xlabel('Model')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.grid(True, alpha=0.3)
        
        # Save figure
        os.makedirs('reports/figures', exist_ok=True)
        plt.savefig('reports/figures/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save comparison results
        comparison_df.to_csv('reports/model_comparison.csv')
        
        return comparison_df


def main():
    """Main function for testing the module"""
    # Load data and model
    from data_preprocessing import DataPreprocessor
    from model_training import ModelTrainer
    
    preprocessor = DataPreprocessor()
    train_df, test_df = preprocessor.load_data()
    processed_data = preprocessor.preprocess_data(train_df, test_df)
    
    # Load a trained model
    model = joblib.load('models/lightgbm.pkl')
    
    # Evaluate model
    evaluator = ModelEvaluator()
    
    # Generate plots and reports
    roc_auc = evaluator.plot_roc_curve(
        model,
        processed_data['X_val'],
        processed_data['y_val'],
        'LightGBM'
    )
    
    cm = evaluator.plot_confusion_matrix(
        model,
        processed_data['X_val'],
        processed_data['y_val'],
        'LightGBM'
    )
    
    feature_importance = evaluator.plot_feature_importance(
        model,
        processed_data['feature_names'],
        'LightGBM'
    )
    
    report = evaluator.generate_classification_report(
        model,
        processed_data['X_val'],
        processed_data['y_val'],
        'LightGBM'
    )
    
    return evaluator, {
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'feature_importance': feature_importance,
        'classification_report': report
    }


if __name__ == "__main__":
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    evaluator, results = main()
