# Credit Card Fraud Detection

Hi, this is my personal project on detecting fraud in credit card transactions using Machine Learning.

I made this to practice handling highly imbalanced data, which is a big challenge in real-world fraud detection.

## Features
- Loads credit card transaction data
- Uses SMOTE to balance the classes (fraud vs normal)
- Trains a RandomForest model
- Evaluates with proper metrics (AUC, recall for fraud class)

## How to use it

1. Clone this repository:
   git clone https://github.com/mohammedaminegoumri/ai-fraud-detection.git

2. Install the packages:
   pip install -r requirements.txt

3. Download the dataset from Kaggle (creditcard.csv) and place it in the data/ folder.

4. Run the script:
   python fraud_detection.py

Let me know if you have any suggestions to improve it!

— Mohammed Amine