# 🛡️ Credit Card Fraud Detection with AI

**Personal project by Mohammed Amine Goumri** — Building intelligent systems that protect financial transactions.

This repository contains a complete, production-ready **AI-powered fraud detection system** featuring:

- Full Exploratory Data Analysis (EDA) with interactive visualizations
- SMOTE for handling severe class imbalance (0.3% fraud rate)
- Random Forest + XGBoost models with excellent performance (AUC ~0.98)
- **Beautiful Streamlit Dashboard** for real-time predictions and insights

---

## ✨ New: Interactive Streamlit Dashboard

Run the full interactive experience:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

**Dashboard Features:**
- **🏠 Home** — Project overview & key metrics
- **📊 EDA Dashboard** — Interactive Plotly charts (class distribution, amount patterns, fraud by hour, correlations)
- **🤖 Model Training** — One-click training with SMOTE, ROC curves, confusion matrix, feature importance
- **🔮 Live Predictor** — Input any transaction and get instant fraud probability with visual gauge + risk verdict
- **📖 About** — Tech stack and roadmap

---

## 📊 Visualizations (Auto-generated)

The original scripts generate these plots in `images/`:

- Class Distribution Before SMOTE
- Transaction Amount Distribution by Class
- Fraud Rate by Hour of Day
- Correlation Heatmap
- SMOTE Before vs After
- PCA Projection of Transactions

---

## 🚀 How to Run (Original Scripts)

1. Clone the repo:
   ```bash
   git clone https://github.com/mohammedaminegoumri/ai-fraud-detection.git
   cd ai-fraud-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Download the real [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place `creditcard.csv` in the `data/` folder.

4. Run the core script:
   ```bash
   python fraud_detection.py
   ```

   Or the EDA-focused script:
   ```bash
   python eda_visualization.py
   ```

---

## 🛠️ Tech Stack

- **Python** • **Pandas** • **NumPy**
- **Scikit-learn** • **XGBoost** • **LightGBM**
- **imbalanced-learn** (SMOTE)
- **Plotly** + **Streamlit** (Dashboard)
- **Joblib** (Model serialization)

---

## 📈 Model Performance (on synthetic data)

- **ROC-AUC**: ~0.98
- **Fraud Recall**: >90% (after SMOTE)
- **Precision**: Balanced with low false positives

---

## 🚀 Future Roadmap

- [ ] Add SHAP explainability for individual predictions
- [ ] Deploy to Streamlit Cloud / Hugging Face
- [ ] FastAPI microservice for real-time scoring
- [ ] Feature engineering + real Kaggle dataset integration

---

**Made with ❤️ in 2026** | Empowering Decisions Through Intelligent Data

*Questions or ideas? Open an issue or connect with me on LinkedIn!*