"""
Data preprocessing module for bank churn prediction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import yaml
import os


class DataPreprocessor:
    """Handles all data preprocessing steps"""
    
    def __init__(self, config_path='config/params.yaml'):
        """
        Initialize preprocessor with configuration
        
        Args:
            config_path (str): Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.preprocessor = None
        self.feature_names = None
        
    def load_data(self, train_path='data/train.csv', test_path='data/test.csv'):
        """
        Load train and test datasets
        
        Args:
            train_path (str): Path to training data
            test_path (str): Path to test data
            
        Returns:
            tuple: (train_df, test_df)
        """
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        print(f"Training data shape: {train_df.shape}")
        print(f"Test data shape: {test_df.shape}")
        print(f"Columns: {train_df.columns.tolist()}")
        
        return train_df, test_df
    
    def preprocess_data(self, train_df, test_df):
        """
        Preprocess the data
        
        Args:
            train_df (pd.DataFrame): Training data
            test_df (pd.DataFrame): Test data
            
        Returns:
            tuple: Preprocessed train and test data
        """
        # Separate features and target
        X_train = train_df.drop('Exited', axis=1)
        y_train = train_df['Exited']
        X_test = test_df.copy()
        
        # Define preprocessing pipeline
        numeric_features = self.config['preprocessing']['numerical_features']
        categorical_features = self.config['preprocessing']['categorical_features']
        
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(drop='first', sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'  # Drop features not specified
        )
        
        # Fit and transform training data
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Get feature names after transformation
        numeric_feature_names = numeric_features
        categorical_feature_names = self.preprocessor.named_transformers_['cat']\
            .named_steps['onehot'].get_feature_names_out(categorical_features)
        self.feature_names = list(numeric_feature_names) + list(categorical_feature_names)
        
        # Split training data for validation
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train_processed, y_train,
            test_size=self.config['model']['test_size'],
            random_state=self.config['model']['random_state'],
            stratify=y_train
        )
        
        print(f"Processed training data shape: {X_train_split.shape}")
        print(f"Validation data shape: {X_val.shape}")
        print(f"Processed test data shape: {X_test_processed.shape}")
        
        return {
            'X_train': X_train_split,
            'X_val': X_val,
            'y_train': y_train_split,
            'y_val': y_val,
            'X_test': X_test_processed,
            'feature_names': self.feature_names
        }
    
    def save_preprocessor(self, path='models/preprocessor.pkl'):
        """
        Save the fitted preprocessor
        
        Args:
            path (str): Path to save the preprocessor
        """
        if self.preprocessor is not None:
            joblib.dump(self.preprocessor, path)
            print(f"Preprocessor saved to {path}")
        else:
            raise ValueError("Preprocessor has not been fitted yet")
    
    def load_preprocessor(self, path='models/preprocessor.pkl'):
        """
        Load a saved preprocessor
        
        Args:
            path (str): Path to the saved preprocessor
            
        Returns:
            The loaded preprocessor
        """
        self.preprocessor = joblib.load(path)
        return self.preprocessor


def main():
    """Main function for testing the module"""
    preprocessor = DataPreprocessor()
    train_df, test_df = preprocessor.load_data()
    processed_data = preprocessor.preprocess_data(train_df, test_df)
    preprocessor.save_preprocessor()
    
    return processed_data


if __name__ == "__main__":
    processed_data = main()
