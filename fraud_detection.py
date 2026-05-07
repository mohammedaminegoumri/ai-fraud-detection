# Credit Card Fraud Detection
# Personal Project by Mohammed Amine

# This script trains a model to detect fraudulent credit card transactions

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')

print('=== Fraud Detection System Starting ===')

def load_data(filepath='data/creditcard.csv'):
    try:
        df = pd.read_csv(filepath)
        print(f'Data loaded successfully. Shape: {df.shape}')
        fraud_count = df['Class'].sum()
        print(f'Fraud transactions: {fraud_count} | Normal: {len(df) - fraud_count}')
        return df
    except FileNotFoundError:
        print('Error: creditcard.csv not found in data/ folder')
        print('Download it from Kaggle: https://www.kaggle.com/mlg-ulb/creditcardfraud')
        return None

def preprocess_data(df):
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def train_and_evaluate(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Handle imbalance
    print('Balancing data with SMOTE...')
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # Train the model
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train_res, y_train_res)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print('\n=== Model Results ===')
    print(classification_report(y_test, y_pred))
    print(f'ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}')
    
    return model

def main():
    df = load_data()
    if df is None:
        return
    
    X, y = preprocess_data(df)
    model = train_and_evaluate(X, y)
    
    print('\nModel trained successfully!')
    print('TODO: Save model using joblib')
    print('TODO: Build a function to predict new transactions')

if __name__ == '__main__':
    main()
