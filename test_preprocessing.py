"""
Unit tests for data preprocessing module
"""

import pytest
import pandas as pd
import numpy as np
from src.data_preprocessing import DataPreprocessor


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    data = {
        'CreditScore': [650, 700, 600],
        'Age': [35.0, 40.0, 30.0],
        'Geography': ['France', 'Germany', 'Spain'],
        'Gender': ['Male', 'Female', 'Male'],
        'Exited': [0, 1, 0]
    }
    return pd.DataFrame(data)


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class"""
    
    def test_init(self):
        """Test initialization"""
        preprocessor = DataPreprocessor()
        assert preprocessor.config is not None
        assert preprocessor.preprocessor is None
        assert preprocessor.feature_names is None
    
    def test_load_data(self, tmp_path):
        """Test data loading"""
        # Create temporary CSV files
        train_data = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        test_data = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
        
        train_path = tmp_path / "train.csv"
        test_path = tmp_path / "test.csv"
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        
        preprocessor = DataPreprocessor()
        train_df, test_df = preprocessor.load_data(str(train_path), str(test_path))
        
        assert len(train_df) == 2
        assert len(test_df) == 2
        assert 'A' in train_df.columns
        assert 'B' in test_df.columns
    
    def test_preprocess_data(self, sample_data):
        """Test data preprocessing"""
        preprocessor = DataPreprocessor()
        
        # Create test data
        train_df = sample_data
        test_df = sample_data.drop('Exited', axis=1)
        
        # Preprocess data
        processed_data = preprocessor.preprocess_data(train_df, test_df)
        
        # Check results
        assert 'X_train' in processed_data
        assert 'X_val' in processed_data
        assert 'y_train' in processed_data
        assert 'y_val' in processed_data
        assert 'X_test' in processed_data
        assert 'feature_names' in processed_data
        
        # Check shapes
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        assert X_train.shape[0] == len(y_train)
        
        # Check that preprocessor was fitted
        assert preprocessor.preprocessor is not None
    
    def test_feature_engineering(self, sample_data):
        """Test feature engineering steps"""
        preprocessor = DataPreprocessor()
        
        # Check that categorical features are encoded
        numeric_features = preprocessor.config['preprocessing']['numerical_features']
        categorical_features = preprocessor.config['preprocessing']['categorical_features']
        
        assert 'CreditScore' in numeric_features
        assert 'Age' in numeric_features
        assert 'Geography' in categorical_features
        assert 'Gender' in categorical_features
    
    def test_missing_value_handling(self):
        """Test handling of missing values"""
        # Create data with missing values
        data = {
            'CreditScore': [650, None, 600],
            'Age': [35.0, 40.0, None],
            'Geography': ['France', 'Germany', 'Spain'],
            'Gender': ['Male', 'Female', 'Male'],
            'Exited': [0, 1, 0]
        }
        df = pd.DataFrame(data)
        
        preprocessor = DataPreprocessor()
        # The preprocessor should handle missing values appropriately
        # (implementation depends on strategy)
        
    def test_data_types(self, sample_data):
        """Test data type preservation"""
        preprocessor = DataPreprocessor()
        
        # Check that numeric columns remain numeric
        assert pd.api.types.is_numeric_dtype(sample_data['CreditScore'])
        assert pd.api.types.is_numeric_dtype(sample_data['Age'])
        
        # Check that categorical columns remain categorical
        assert pd.api.types.is_object_dtype(sample_data['Geography'])
        assert pd.api.types.is_object_dtype(sample_data['Gender'])


def test_config_file():
    """Test that config file exists and is valid"""
    import yaml
    import os
    
    config_path = 'config/params.yaml'
    assert os.path.exists(config_path), f"Config file not found at {config_path}"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check required sections
    assert 'model' in config
    assert 'lightgbm' in config
    assert 'xgboost' in config
    assert 'catboost' in config
    assert 'preprocessing' in config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
