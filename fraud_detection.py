# Credit Card Fraud Detection
# Personal Project by Mohammed Amine

# This script trains a model to detect fraudulent credit card transactions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import os
import warnings

warnings.filterwarnings('ignore')

print('=== Fraud Detection System Starting ===')

# Create images folder if it doesn't exist
def create_images_folder():
    os.makedirs('images', exist_ok=True)
    print('✅ images/ folder created for visualizations')

def generate_eda_plots(df):
    print('\n📊 Generating EDA visualizations...')
    create_images_folder()
    
    # 1. Class distribution before SMOTE
    plt.figure(figsize=(8,5))
    class_counts = df['Class'].value_counts()
    plt.bar(['Legit (0)', 'Fraud (1)'], class_counts.values, color=['#1f77b4', '#ff7f0e'])
    plt.title('Class Distribution Before SMOTE\n(Highly Imbalanced)', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Transactions')
    for i, v in enumerate(class_counts.values):
        plt.text(i, v + 1000, str(v), ha='center', fontsize=11)
    plt.savefig('images/class_distribution_before_smote.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # 2. Transaction Amount Distribution
    plt.figure(figsize=(9,5.5))
    sns.histplot(data=df, x='Amount', hue='Class', bins=60, multiple='stack', palette=['#1f77b4', '#ff7f0e'], alpha=0.85)
    plt.title('Transaction Amount Distribution by Class', fontsize=14, fontweight='bold')
    plt.xlabel('Transaction Amount ($)') 
    plt.ylabel('Count')
    plt.savefig('images/transaction_amount_distribution.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print('✅ EDA plots saved to images/ folder')

def generate_smote_comparison():
    print('📈 Generating SMOTE comparison plot...')
    # Simulated before vs after (since real data is huge)
    plt.figure(figsize=(10,5))
    
    # Before SMOTE
    plt.subplot(1, 2, 1)
    plt.bar(['Legit', 'Fraud'], [284315, 492], color=['#1f77b4', '#ff7f0e'])
    plt.title('Before SMOTE')
    plt.ylabel('Count')
    
    # After SMOTE
    plt.subplot(1, 2, 2)
    plt.bar(['Legit', 'Fraud'], [284315, 284315], color=['#1f77b4', '#ff7f0e'])
    plt.title('After SMOTE (Balanced)')
    plt.tight_layout()
    plt.savefig('images/smote_before_after.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('✅ SMOTE comparison plot saved')

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
    
    # Generate visualizations
    generate_eda_plots(df)
    generate_smote_comparison()
    
    X, y = preprocess_data(df)
    model = train_and_evaluate(X, y)
    
    print('\n🎉 Model trained successfully!')
    print('📁 Check the "images/" folder for EDA and SMOTE plots')
    print('TODO: Save model using joblib')

if __name__ == "__main__":
    main()
