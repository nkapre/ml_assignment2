import time
import pandas as pd
import numpy as np
from app import preprocess_data
from models.xg_boost_model import run_xgboost
from sklearn.metrics import accuracy_score

def measure_performance(file_path):
    print(f"--- Performance Report: Churn Kernel Inc. ---")
    
    # 1. Measure Data Loading
    start_time = time.time()
    data = pd.read_csv(file_path)
    load_time = time.time() - start_time
    print(f"Step 1: Data Loading       | {load_time:.4f} seconds")

    # 2. Measure Preprocessing
    start_time = time.time()
    processed_df = preprocess_data(data)
    X = processed_df.drop(columns=['Churn'])
    y = processed_df['Churn']
    preprocess_time = time.time() - start_time
    print(f"Step 2: Preprocessing      | {preprocess_time:.4f} seconds")

    # 3. Measure Model Training & Prediction
    # run_xgboost handles both internal fit and predict
    start_time = time.time()
    y_pred, y_prob = run_xgboost(X, y)
    execution_time = time.time() - start_time
    
    # Calculate approximate time per tree
    # Accessing default n_estimators=50 from CustomGradientBoost
    time_per_tree = execution_time / 50 
    print(f"Step 3: Training & Predict | {execution_time:.4f} seconds")
    print(f"        (Avg. per Tree)    | {time_per_tree:.4f} seconds")

    # 4. Final Results
    accuracy = accuracy_score(y, y_pred)
    print(f"---")
    print(f"Final Model Accuracy: {accuracy:.4f}")
    print(f"Total Processing Time: {load_time + preprocess_time + execution_time:.4f} seconds")

if __name__ == "__main__":
    # Replace with your actual dataset path
    dataset_path = '/Users/nikhilkapre/Documents/assignments/ml/assignment-2/WA_Fn-UseC_-Telco-Customer-Churn.csv' 
    try:
        measure_performance(dataset_path)
    except FileNotFoundError:
        print(f"Error: Please place your dataset at {dataset_path}")