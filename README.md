# Credit Card Fraud Detection with AI

Hey, this is **my** personal project where I built a Machine Learning system to detect fraudulent credit card transactions.

I worked on handling heavily imbalanced data using **SMOTE** and did proper Exploratory Data Analysis (EDA).

## 📊 Visualizations

Here are the plots I generated:

![Class Distribution Before SMOTE](images/class_distribution_before_smote.png)

![Transaction Amount Distribution by Class](images/transaction_amount_distribution.png)

![Fraud Rate by Hour of Day](images/fraud_by_hour.png)

![Correlation Heatmap](images/correlation_heatmap.png)

![SMOTE Before vs After](images/smote_before_after.png)

![PCA Projection of Transactions](images/pca_projection.png)

## What I did:
- Full EDA with multiple visualizations (Class distribution, Amount, Time patterns, Correlation)
- Applied **SMOTE** to balance the classes
- Used PCA for dimensionality reduction visualization
- Trained a Random Forest model with good performance

## How to run it

1. Clone the repo:
   ```bash
   git clone https://github.com/mohammedaminegoumri/ai-fraud-detection.git
   cd ai-fraud-detection
   ```

2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. (Recommended) Download the full dataset from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and put `creditcard.csv` inside a `data/` folder.

4. Run the script:
   ```bash
   python fraud_detection.py
   ```

The script will automatically create the `images/` folder and save all the plots shown above.

Let me know if you want me to add a **Streamlit web app** or a FastAPI endpoint next!

— Mohammed Amine