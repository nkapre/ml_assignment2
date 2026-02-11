import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import metrics from sklearn for the evaluation dashboard
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef, confusion_matrix)

# Import the custom "from scratch" functions
from models.logistic_regression import run_logistic_regression
from models.decision_tree import run_decision_tree
from models.knn import run_knn
from models.naive_bayes import run_naive_bayes
from models.random_forest import run_random_forest
from models.xg_boost_model import run_xgboost

st.set_page_config(page_title="Churn Classification (From Scratch)", layout="wide")

st.title("📊 Customer Churn Classification - Custom Implementations")
st.markdown("This application uses machine learning models built from scratch without scikit-learn's estimator classes.")

# Sidebar Configuration
st.sidebar.header("Upload & Model Settings")
uploaded_file = st.sidebar.file_uploader("Upload Test CSV (with 'Churn' column)", type="csv")
model_choice = st.sidebar.selectbox("Select ML Model", 
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"])

# Helper function to process data
def preprocess_data(df):
    # Drop ID and handle the target
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    
    # Map Churn to binary
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Simple encoding for categorical variables
    df = pd.get_dummies(df)
    
    # Fill missing values if any
    df = df.fillna(0)
    return df

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Data Preview", data.head(5))
    
    processed_df = preprocess_data(data)
    
    if 'Churn' in processed_df.columns:
        X = processed_df.drop(columns=['Churn'])
        y = processed_df['Churn']
        
        # Mapping model selection to the imported scratch functions
        model_functions = {
            "Logistic Regression": run_logistic_regression,
            "Decision Tree": run_decision_tree,
            "KNN": run_knn,
            "Naive Bayes": run_naive_bayes,
            "Random Forest": run_random_forest,
            "XGBoost": run_xgboost
        }
        
        with st.spinner(f"Training and evaluating {model_choice}..."):
            # Execute the custom model
            # These return (y_pred, y_prob) or (y_pred, y_pred)
            y_pred, y_prob = model_functions[model_choice](X, y)
        
        # Calculation of Metrics
        metrics = {
            "Accuracy": accuracy_score(y, y_pred),
            "AUC Score": roc_auc_score(y, y_prob),
            "Precision": precision_score(y, y_pred, zero_division=0),
            "Recall": recall_score(y, y_pred, zero_division=0),
            "F1 Score": f1_score(y, y_pred, zero_division=0),
            "MCC Score": matthews_corrcoef(y, y) if len(np.unique(y)) > 1 else 0
        }

        # Dashboard Layout
        st.divider()
        st.subheader(f"Evaluation Metrics: {model_choice}")
        
        m_cols = st.columns(6)
        for i, (label, value) in enumerate(metrics.items()):
            m_cols[i].metric(label, f"{value:.4f}")

        # Confusion Matrix and Report
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("#### Confusion Matrix")
            cm = confusion_matrix(y, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', ax=ax)
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            st.pyplot(fig)
            
        with col2:
            st.write("#### Prediction Distribution")
            dist_df = pd.DataFrame({'Actual': y, 'Predicted': y_pred})
            st.bar_chart(dist_df.apply(pd.Series.value_counts))
            
    else:
        st.error("Error: The dataset must contain a 'Churn' column for evaluation.")

else:
    st.info("Please upload a CSV file to begin.")