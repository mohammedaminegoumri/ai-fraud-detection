import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
from datetime import datetime

st.set_page_config(page_title="AI Fraud Detection", page_icon="🛡️", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main {padding-top: 2rem;}
    .stApp {
        background: linear-gradient(135deg, #0f0f23, #1a1a2e);
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ AI Fraud Detection Dashboard")
st.markdown("**Real-time Credit Card Fraud Detection using Machine Learning**")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", ["Home", "EDA", "Model Training", "Live Predictor", "About"])

if page == "Home":
    st.subheader("Welcome to the Fraud Detection System")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Transactions", "284,807", "+12.4%")
    with col2:
        st.metric("Fraud Cases", "492", "-8.2%")
    with col3:
        st.metric("Fraud Rate", "0.17%", "-0.03%")
    
    st.info("This dashboard demonstrates an end-to-end AI system for detecting fraudulent credit card transactions.")

elif page == "EDA":
    st.subheader("Exploratory Data Analysis")
    st.write("Sample synthetic data loaded")
    # Generate synthetic data for demo
    np.random.seed(42)
    n = 1000
    data = pd.DataFrame({
        'Time': np.random.randint(0, 175000, n),
        'Amount': np.random.lognormal(3, 1, n),
        'Class': np.random.choice([0,1], n, p=[0.95, 0.05])
    })
    st.dataframe(data.head(), use_container_width=True)
    
    fig = px.pie(data, names='Class', title='Fraud vs Legitimate Transactions')
    st.plotly_chart(fig, use_container_width=True)

elif page == "Model Training":
    st.subheader("Train Fraud Detection Model")
    if st.button("🚀 Train Random Forest Model"):
        with st.spinner("Training model..."):
            # Synthetic training
            X = np.random.rand(1000, 10)
            y = np.random.choice([0,1], 1000, p=[0.9, 0.1])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
            
            st.success(f"Model trained successfully! ROC-AUC: **{auc:.4f}**")
            st.text(classification_report(y_test, y_pred))
            
            # Save model
            joblib.dump(model, 'fraud_model.pkl')
            st.balloons()

elif page == "Live Predictor":
    st.subheader("🔮 Live Transaction Fraud Prediction")
    
    col1, col2 = st.columns([1,1])
    with col1:
        amount = st.slider("Transaction Amount ($)", 0.0, 5000.0, 75.0)
        time = st.slider("Time (seconds since first transaction)", 0, 175000, 50000)
    with col2:
        v_mean = st.slider("Feature Vector Mean (simplified)", -2.0, 2.0, 0.0)
    
    if st.button("Analyze Transaction", type="primary"):
        # Simple mock prediction
        fraud_prob = 0.12 + (amount > 2000)*0.6 + np.random.rand()*0.1
        fraud_prob = min(fraud_prob, 0.98)
        
        if fraud_prob > 0.7:
            st.error(f"🚨 HIGH RISK - Fraud Probability: **{fraud_prob:.1%}**")
        elif fraud_prob > 0.3:
            st.warning(f"⚠️ Suspicious - Fraud Probability: **{fraud_prob:.1%}**")
        else:
            st.success(f"✅ Legitimate - Fraud Probability: **{fraud_prob:.1%}**")
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = fraud_prob * 100,
            title = {'text': "Fraud Risk"},
            gauge = {'axis': {'range': [0, 100]},
                    'bar': {'color': "red" if fraud_prob > 0.5 else "green"},
                    'steps': [
                        {'range': [0, 30], 'color': "green"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ]}))
        st.plotly_chart(fig, use_container_width=True)

elif page == "About":
    st.subheader("About this Project")
    st.write("Built with ❤️ by Mohammed Amine Goumri")
    st.write("This is a modern Streamlit dashboard showcasing AI-powered fraud detection.")

st.caption("© 2026 Mohammed Amine Goumri | Data Analyst & BI Developer")
