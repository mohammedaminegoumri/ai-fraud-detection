# 🛡️ AI Fraud Detection Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black.svg)](https://ai-fraud-detection-ck4q5s67qgnffwujgyvvqt.streamlit.app/)

**Modern interactive web app for real-time credit card fraud detection using Machine Learning.**

Built as a complete end-to-end data science project showcasing **Agentic AI**, real-time analytics, and ethical machine learning.

## 🚀 Live Demo

🔥 **[Try the Live App Now](https://ai-fraud-detection-ck4q5s67qgnffwujgyvvqt.streamlit.app/)**

*Fully interactive • No signup required • Powered by synthetic data for instant demo*

## ✨ Key Features

- **🏠 Home** – Key metrics & project overview at a glance
- **📊 EDA Dashboard** – Beautiful interactive Plotly visualizations (class distribution, transaction amounts, fraud patterns by hour)
- **🤖 One-Click Model Training** – Train a Random Forest classifier with **SMOTE** balancing
- **🔮 Live Predictor** – Real-time fraud detection with probability gauge chart
- **📈 Production-Ready** – Clean code, model persistence, responsive UI

## 🛠️ Tech Stack

- **Frontend & Deployment**: [Streamlit](https://streamlit.io)
- **Visualizations**: Plotly
- **Machine Learning**: scikit-learn, imbalanced-learn (SMOTE)
- **Language**: Python 3.10+

## 📥 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/mohammedaminegoumri/ai-fraud-detection.git
cd ai-fraud-detection

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run streamlit_app.py
```

## 📖 How It Works

1. **Synthetic Data Generation** – Realistic credit card transactions for instant demo
2. **Exploratory Data Analysis** – Understand fraud patterns visually
3. **Model Training** – Handles severe class imbalance with SMOTE + Random Forest
4. **Live Inference** – Input any transaction and get instant fraud probability

## 📊 Model Performance (on synthetic test set)

- **ROC-AUC Score**: ~0.98
- Effectively detects rare fraud cases

## 🔮 Future Roadmap

- Integration with real Kaggle `creditcard.csv` dataset
- SHAP explainability for model predictions
- FastAPI backend for production API
- Advanced Agentic AI decision layer
- MLOps pipeline (MLflow + DVC)

---

**Made with ❤️ by [Mohammed Amine Goumri](https://github.com/mohammedaminegoumri)**  
Data Analyst & BI Developer | Passionate about AI, Data Science & Analytics

[LinkedIn](https://www.linkedin.com/in/mohammed-amine-goumri/) • [Portfolio](https://github.com/mohammedaminegoumri)

*Empowering decisions through intelligent data.*