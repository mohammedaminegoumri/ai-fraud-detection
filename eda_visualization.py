# EDA and SMOTE Visualization for Fraud Detection
# Personal Project by Mohammed Amine

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# Create images folder
os.makedirs('images', exist_ok=True)
print('=== Generating EDA and SMOTE Visualizations ===')

def load_data():
    try:
        df = pd.read_csv('data/creditcard.csv')
        print(f'Data loaded: {df.shape[0]:,} transactions')
        return df
    except FileNotFoundError:
        print('⚠️  creditcard.csv not found in data/ folder.')
        print('You can download it from Kaggle: https://www.kaggle.com/mlg-ulb/creditcardfraud')
        return None

def generate_eda_plots(df):
    print('Generating EDA plots...')
    
    # 1. Class Distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Class', data=df)
    plt.title('Class Distribution (0 = Normal, 1 = Fraud)', fontsize=14)
    plt.xlabel('Class')
    plt.ylabel('Number of Transactions')
    plt.savefig('images/class_distribution.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # 2. Transaction Amount Distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='Amount', hue='Class', bins=50, kde=True)
    plt.title('Transaction Amount Distribution by Class', fontsize=14)
    plt.xlabel('Amount ($)') 
    plt.yscale('log')
    plt.savefig('images/amount_distribution.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # 3. Correlation Heatmap (first 15 features for clarity)
    plt.figure(figsize=(12, 10))
    corr = df.corr()
    sns.heatmap(corr, cmap='coolwarm', annot=False)
    plt.title('Feature Correlation Heatmap', fontsize=14)
    plt.savefig('images/correlation_heatmap.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print('✅ EDA plots saved to images/ folder')

def show_smote_effect(df):
    print('Applying SMOTE and visualizing effect...')
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Before SMOTE
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.countplot(x=y)
    plt.title('Before SMOTE\nImbalanced Data')
    plt.ylabel('Count')
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    plt.subplot(1, 2, 2)
    sns.countplot(x=y_res)
    plt.title('After SMOTE\nBalanced Data')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('images/smote_effect.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print('✅ SMOTE before/after plot saved!')
    print(f'   Original fraud cases: {y.sum()}')
    print(f'   After SMOTE: {y_res.sum()}')

def main():
    df = load_data()
    if df is None:
        return
        
    generate_eda_plots(df)
    show_smote_effect(df)
    
    print('\n🎉 All visualizations completed!')
    print('Check the "images/" folder')

if __name__ == "__main__":
    main()
