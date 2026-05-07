# AI Fraud Detection System

🚀 Machine Learning model to detect **credit card fraud** with high precision.

## Features
- Handles highly imbalanced dataset (fraud is ~0.17%)
- Multiple ML models (Random Forest, XGBoost, LightGBM)
- Advanced evaluation metrics (Precision, Recall, F1, AUC-PR)
- Feature importance analysis

## Dataset
We use the famous [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset from Kaggle.

## Project Structure
```
ai-fraud-detection/
├── README.md
├── requirements.txt
├── data/
├── notebooks/
├── src/
│   ├── data_preprocessing.py
│   ├── train.py
│   └── evaluate.py
└── models/
```