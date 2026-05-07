# ================================================
# 🚀 AI-Powered Credit Card Fraud Detection
# ================================================
# Author: Built with Grok
# Goal: Detect fraudulent transactions with high precision

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, average_precision_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

print("🚀 AI Fraud Detection System Initialized!\n")

def load_data(filepath='data/creditcard.csv'):
    """Load the credit card fraud dataset"""
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Data loaded successfully! Shape: {df.shape}")
        print(f"Fraud cases: {df['Class'].sum()} | Normal: {len(df)-df['Class'].sum()}")
        return df
    except FileNotFoundError:
        print("❌ Data file not found. Please download creditcard.csv from Kaggle")
        print("Link: https://www.kaggle.com/mlg-ulb/creditcardfraud")
        return None

def preprocess_data(df):
    """Preprocess the data"""
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def train_model(X, y):
    """Train model with SMOTE for imbalance"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Handle imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    print(f"After SMOTE - Training samples: {len(X_train_res)}")
    
    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train_res, y_train_res)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluation
    print("\n📊 Model Performance:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"PR-AUC Score: {average_precision_score(y_test, y_pred_proba):.4f}")
    
    return model, X_test, y_test, y_pred_proba

def main():
    print("Starting Fraud Detection Pipeline...\n")
    df = load_data()
    if df is None:
        return
    
    X, y, scaler = preprocess_data(df)
    model, X_test, y_test, y_pred_proba = train_model(X, y)
    
    print("\n✅ Model training completed!")
    print("Ready for deployment or further improvement (XGBoost, Neural Nets, SHAP explainability, etc.)")

if __name__ == "__main__":
    main()
