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
from sklearn.decomposition import PCA
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
    
    # 3. Correlation Heatmap
    plt.figure(figsize=(12,10))
    corr = df.corr()
    sns.heatmap(corr, cmap='coolwarm', annot=False, linewidths=0.5)
    plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('images/correlation_heatmap.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # 4. Fraud by Hour of Day
    df['Hour'] = (df['Time'] % 86400) // 3600
    plt.figure(figsize=(10,6))
    hourly_fraud = df.groupby('Hour')['Class'].mean() * 100
    plt.plot(hourly_fraud.index, hourly_fraud.values, marker='o', color='#ff7f0e', linewidth=2)
    plt.title('Fraud Rate by Hour of Day (%)', fontsize=14, fontweight='bold')
    plt.xlabel('Hour of Day (0-23)')
    plt.ylabel('Fraud Rate (%)')
    plt.grid(True, alpha=0.3)
    plt.savefig('images/fraud_by_hour.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print('✅ EDA plots saved to images/ folder')

def generate_smote_comparison():
    print('📈 Generating SMOTE comparison plot...')
    create_images_folder()
    
    plt.figure(figsize=(11,5))
    
    # Before SMOTE
    plt.subplot(1, 2, 1)
    plt.bar(['Legit', 'Fraud'], [284315, 492], color=['#1f77b4', '#ff7f0e'])
    plt.title('Before SMOTE')
    plt.ylabel('Count')
    plt.ylim(0, 300000)
    
    # After SMOTE
    plt.subplot(1, 2, 2)
    plt.bar(['Legit', 'Fraud'], [284315, 284315], color=['#1f77b4', '#ff7f0e'])
    plt.title('After SMOTE (Balanced)')
    plt.ylabel('Count')
    plt.ylim(0, 300000)
    plt.tight_layout()
    plt.savefig('images/smote_before_after.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('✅ SMOTE comparison plot saved')

def generate_pca_plot(df):
    print('🔍 Generating PCA visualization...')
    create_images_folder()
    
    sample_df = df.sample(frac=0.1, random_state=42) if len(df) > 100000 else df
    
    X = sample_df.drop(['Class', 'Time', 'Amount'], axis=1, errors='ignore')
    y = sample_df['Class']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(10,7))
    plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1], alpha=0.5, label='Legit', color='#1f77b4', s=8)
    plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1], alpha=0.9, label='Fraud', color='#ff7f0e', s=20)
    plt.title('PCA Projection of Transactions\n(Fraud cases highlighted)', fontsize=14, fontweight='bold')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('images/pca_projection.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('✅ PCA plot saved')

def load_data(filepath='data/creditcard.csv'):
    try:
        df = pd.read_csv(filepath)
        print(f'Data loaded successfully. Shape: {df.shape}')
        fraud_count = df['Class'].sum()
        print(f'Fraud transactions: {fraud_count} | Normal: {len(df) - fraud_count}')
        return df
    except FileNotFoundError:
        print('⚠️  creditcard.csv not found in data/ folder')
        print('Download it from Kaggle: https://www.kaggle.com/mlg-ulb/creditcardfraud')
        return None

def preprocess_data(df):
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print('Balancing data with SMOTE...')
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train_res, y_train_res)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print('\n=== Model Results ===')
    print(classification_report(y_test, y_pred))
    print(f'ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}')
    
    return model

def main():
    df = load_data()
    if df is None:
        print('\n💡 Tip: Add the dataset to continue with full EDA & training')
        return
    
    generate_eda_plots(df)
    generate_smote_comparison()
    generate_pca_plot(df)
    
    X, y = preprocess_data(df)
    model = train_and_evaluate(X, y)
    
    print('\n🎉 Everything completed successfully!')
    print('📁 Check the "images/" folder for all EDA plots')
    print('TODO: Save trained model using joblib')

if __name__ == "__main__":
    main()
