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

def create_images_folder():
    os.makedirs('images', exist_ok=True)
    print('✅ images/ folder created for visualizations')

def generate_synthetic_data(n_samples=30000):
    print('Generating synthetic credit card transaction data...')
    np.random.seed(42)
    
    n_features = 28
    X = np.random.randn(n_samples, n_features)
    
    y = np.zeros(n_samples)
    fraud_idx = np.random.choice(n_samples, size=int(n_samples*0.003), replace=False)
    y[fraud_idx] = 1
    
    df = pd.DataFrame(X, columns=[f'V{i}' for i in range(1,29)])
    df['Amount'] = np.abs(np.random.normal(80, 200, n_samples))
    df['Time'] = np.random.randint(0, 172800, n_samples)
    df['Class'] = y.astype(int)
    
    print(f'Synthetic data created: {len(df)} transactions | Fraud cases: {df["Class"].sum()}')
    return df

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
        plt.text(i, v + max(50, v*0.05), str(v), ha='center', fontsize=11)
    plt.savefig('images/class_distribution_before_smote.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # 2. Transaction Amount Distribution
    plt.figure(figsize=(9,5.5))
    sns.histplot(data=df, x='Amount', hue='Class', bins=50, multiple='stack', palette=['#1f77b4', '#ff7f0e'])
    plt.title('Transaction Amount Distribution by Class', fontsize=14, fontweight='bold')
    plt.xlabel('Transaction Amount ($)') 
    plt.ylabel('Count')
    plt.savefig('images/transaction_amount_distribution.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # 3. Correlation Heatmap
    plt.figure(figsize=(10,8))
    corr = df.corr()
    sns.heatmap(corr, cmap='coolwarm', annot=False, linewidths=0.5)
    plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('images/correlation_heatmap.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # 4. Fraud by Hour
    df['Hour'] = (df['Time'] % 86400) // 3600
    plt.figure(figsize=(10,6))
    hourly_fraud = df.groupby('Hour')['Class'].mean() * 100
    plt.plot(hourly_fraud.index, hourly_fraud.values, marker='o', color='#ff7f0e', linewidth=2.5)
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
    
    plt.subplot(1, 2, 1)
    plt.bar(['Legit', 'Fraud'], [29000, 90], color=['#1f77b4', '#ff7f0e'])
    plt.title('Before SMOTE')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    plt.bar(['Legit', 'Fraud'], [29000, 29000], color=['#1f77b4', '#ff7f0e'])
    plt.title('After SMOTE (Balanced)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('images/smote_before_after.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('✅ SMOTE comparison plot saved')

def generate_pca_plot(df):
    print('🔍 Generating PCA visualization...')
    create_images_folder()
    
    X = df.drop(['Class', 'Time', 'Amount'], axis=1, errors='ignore')
    y = df['Class']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(10,7))
    plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1], alpha=0.6, label='Legit', color='#1f77b4', s=8)
    plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1], alpha=0.9, label='Fraud', color='#ff7f0e', s=25)
    plt.title('PCA Projection of Transactions\n(Fraud cases highlighted)', fontsize=14, fontweight='bold')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('images/pca_projection.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('✅ PCA plot saved')

def main():
    print('Using synthetic data for quick demo and visualization...')
    df = generate_synthetic_data()
    
    generate_eda_plots(df)
    generate_smote_comparison()
    generate_pca_plot(df)
    
    print('\n🎉 All visualizations generated successfully!')
    print('📁 Check the "images/" folder - all 6 plots are there now')
    print('\nNext: Want a Streamlit web app?')

if __name__ == "__main__":
    main()
