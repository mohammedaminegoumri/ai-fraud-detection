# Credit Card Fraud Detection with AI

Hey, this is **my** personal project where I built a Machine Learning system to detect fraudulent credit card transactions.

Real fraud datasets are extremely imbalanced, so I practiced using **SMOTE** and proper EDA.

## What I added recently:
- Full **Exploratory Data Analysis (EDA)** visualizations
- **SMOTE** before/after comparison plots
- Clean, well-commented code

## Visualizations (Images)
When you run the script, it automatically creates an `images/` folder containing:

- `class_distribution_before_smote.png` → Shows how unbalanced the original data is
- `smote_before_after.png` → Before vs After applying SMOTE
- `transaction_amount_distribution.png`
- `correlation_heatmap.png`

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

3. Download the dataset from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and put `creditcard.csv` inside a `data/` folder.

4. Run the main script (this will generate the images + train the model):
   ```bash
   python fraud_detection.py
   ```

The images will be saved in the `images/` folder.

Let me know what you think or if you want me to add a web app / Streamlit version next!

— Mohammed Amine