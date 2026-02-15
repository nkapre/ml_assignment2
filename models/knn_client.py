import pandas as pd
import time
from knn import CustomKNN

def run_knn_client(dataset_path):
    try:
        # 1. Load the dataset
        print(f"Loading dataset from: {dataset_path}...")
        df = pd.read_csv(dataset_path)
        df_numeric = pd.get_dummies(df)
        
        # Assuming the last column is the label and the rest are features
        #X = df.iloc[:, :-1].values
        #y = df.iloc[:, -1].values
        X = df_numeric.iloc[:, :-1].values.astype(float) # Ensure float type
        y = df_numeric.iloc[:, -1].values
        
        # 2. Initialize the model with k=5
        knn = CustomKNN(k=5)
        
        # 3. Measure training (fit) time
        start_fit = time.perf_counter()
        knn.fit(X, y)
        end_fit = time.perf_counter()
        
        # 4. Measure prediction time
        start_pred = time.perf_counter()
        predictions = knn.predict(X)
        end_pred = time.perf_counter()
        
        # Output results
        print("\n--- Execution Summary ---")
        print(f"Total Samples: {len(X)}")
        print(f"Fit Time: {end_fit - start_fit:.4f} seconds")
        print(f"Prediction Time: {end_pred - start_pred:.4f} seconds")
        print(f"Total Execution Time: {end_pred - start_fit:.4f} seconds")
        
        return predictions

    except FileNotFoundError:
        print("Error: The specified dataset file was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Replace 'your_data.csv' with your actual dataset path
    dataset_path = '/Users/nikhilkapre/Documents/assignments/ml/assignment-2/WA_Fn-UseC_-Telco-Customer-Churn.csv' 
    run_knn_client(dataset_path)