
# Bank Customer Churn Prediction - Deep Learning Approach

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

A comprehensive machine learning project for predicting bank customer churn using gradient boosting algorithms (LightGBM, XGBoost, CatBoost) and deep learning approaches.
![imresizer-1706204681767.jpg](attachment:imresizer-1706204681767.jpg)
## 📋 Project Overview

This project analyzes customer data from a bank to predict whether customers will churn (leave the bank). The notebook demonstrates:
- **Exploratory Data Analysis (EDA)** of customer demographics and behavior
- **Feature Engineering** and preprocessing techniques
- **Multiple ML Model Training** with LightGBM, XGBoost, and CatBoost
- **Model Evaluation** and performance comparison
- **Churn Rate Analysis** (21.16% in the dataset)

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
Jupyter Notebook/Lab
```
# Model Performance Report

## Executive Summary
This report summarizes the performance of machine learning models trained to predict bank customer churn. The dataset contains 165,034 customer records with a churn rate of 21.16%.

## Models Evaluated
1. **LightGBM** - Gradient boosting framework
2. **XGBoost** - Optimized gradient boosting
3. **CatBoost** - Categorical feature handling

## Performance Metrics

### ROC-AUC Scores
| Model | ROC-AUC | Rank |
|-------|---------|------|
| CatBoost | 0.872 | 1 |
| LightGBM | 0.865 | 2 |
| XGBoost | 0.858 | 3 |

### Detailed Metrics

## 📊 Dataset
- The dataset contains customer information including:

- Demographics: Age, Gender, Geography

- Financial: CreditScore, Balance, EstimatedSalary

- Behavioral: Tenure, NumOfProducts, IsActiveMember

- Target: Exited (1 = Churned, 0 = Stayed)

## Dataset Statistics:

- Training samples: 165,034

- Test samples: 165,034

- Features: 14 (train), 13 (test)

- Churn rate: 21.16%

## 🏗️ Project Structure
```
bank-churn-prediction/
│
├── bank_churn_dataset.ipynb      # Main Jupyter notebook
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── LICENSE                       # License file
├── .gitignore                    # Git ignore file
├── data/                         # Data directory
│   ├── train.csv                 # Training dataset
│   └── test.csv                  # Test dataset
├── models/                       # Saved models
│   ├── lightgbm_model.pkl
│   ├── xgboost_model.pkl
│   └── catboost_model.pkl
├── src/                          # Source code
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── evaluation.py
├── notebooks/                    # Additional notebooks
│   └── exploratory_analysis.ipynb
├── reports/                      # Generated reports
│   └── model_performance.md
├── config/                       # Configuration files
│   └── params.yaml
└── tests/                        # Unit tests
    └── test_preprocessing.py
```
 # 🔧 Implementation Details
## Data Preprocessing
- Handling missing values

- Feature encoding (OneHotEncoder for categorical variables)

- Feature scaling (StandardScaler for numerical variables)

- Train-test split with stratification

## Models Implemented
- LightGBM - Gradient boosting framework by Microsoft

- XGBoost - Optimized gradient boosting library

- CatBoost - Handles categorical features natively

- Deep Learning Model (Planned for future work)

## Evaluation Metrics
- ROC-AUC Score (Primary metric)

- Accuracy, Precision, Recall

- Confusion Matrix Analysis

- Feature Importance Analysis

# 📈 Results
## Performance Summary   
l	ROC-AUC Score	Accuracy	Precision	Recall
LightGBM	0.86	0.85	0.78	0.63
XGBoost	0.85	0.84	0.76	0.61
CatBoost	0.87	0.86	0.79	0.65

## Key Findings
 - Age and Balance are the most important predictors of churn

- German customers have higher churn rates compared to French/Spanish

- Inactive members are more likely to churn

- Customers with 2 products show lowest churn rates
# 📱 Usage
## Running the Complete Pipeline  
# 👨‍💻 Author
## Hammad Zahid

- LinkedIn: https//linkedin.com/in/hammad-zahid-xyz

- GitHub: https//github.com/Hamad-Ansari

- Email: mrhammadzahi24@gmail.com

## 🙏 Acknowledgments
- Dataset sourced from Kaggle Bank Customer Churn Prediction

- Thanks to all contributors and maintainers of the ML libraries used

- Inspired by real-world banking analytics problems

