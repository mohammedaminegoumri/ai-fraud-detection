# 🚀 AI Fraud Detection System

Machine Learning model to **detect and prevent credit card fraud** with high accuracy.

## ✨ Features
- Handles extreme class imbalance (fraud = 0.17%)
- Uses SMOTE + Random Forest (can easily swap to XGBoost/LightGBM)
- Strong focus on **Precision & Recall** (critical for fraud)
- Feature importance analysis

## 📊 Dataset
We are using the popular **Credit Card Fraud Detection** dataset:
→ [Kaggle - Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## 🚀 Quick Start

1. Clone the repo
```bash
git clone https://github.com/mohammedaminegoumri/ai-fraud-detection.git
cd ai-fraud-detection
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Download the dataset from Kaggle and put `creditcard.csv` inside the `data/` folder

4. Train the model
```bash
python src/train.py
```

## Next Steps (we can add them together)
- Try XGBoost / LightGBM
- Add Streamlit web demo
- Model explainability (SHAP)
- API with FastAPI
- Real-time fraud scoring

---
**Ready to start coding?** Just say what you want to do next!