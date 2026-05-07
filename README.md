# Credit Card Fraud Detection

Hi, this is my personal project on detecting fraud in credit card transactions using Machine Learning.

I made this to practice handling highly imbalanced data, which is a big challenge in real-world fraud detection.

## Features
- Loads credit card transaction data
- Uses **SMOTE** to balance the classes (fraud vs normal)
- Trains a RandomForest model
- Full EDA (Exploratory Data Analysis) with visualizations

## Visualizations
I added a script that generates nice plots automatically:

Run this to create the images:
```bash
python eda_visualization.py
```

This will create an `images/` folder with:
- `class_distribution.png` → Shows how imbalanced the data is
- `smote_effect.png` → Before vs After SMOTE
- `amount_distribution.png`
- `correlation_heatmap.png`

## How to use it

1. Clone this repository:
   ```bash
   git clone https://github.com/mohammedaminegoumri/ai-fraud-detection.git
   ```

2. Install the packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from Kaggle (`creditcard.csv`) and put it in the `data/` folder.
   Link: https://www.kaggle.com/mlg-ulb/creditcardfraud

4. Generate visualizations:
   ```bash
   python eda_visualization.py
   ```

5. Train the model:
   ```bash
   python fraud_detection.py
   ```

Let me know if you have any suggestions to improve it!

— Mohammed Amine