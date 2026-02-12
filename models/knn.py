import numpy as np

class CustomKNN:
    def __init__(self, k=5):
        """
        Optimized K-Nearest Neighbors Classifier.
        Uses vectorized matrix operations for distance calculation to improve performance.
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Stores the training data.
        """
        # Ensure data is in numpy format for fast computation
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        """
        Predicts labels for the provided dataset using vectorized distance calculations.
        """
        X_test = np.array(X)
        
        # Optimization: Calculate Squared Euclidean Distance using Matrix Multiplication
        # Formula: ||A - B||^2 = ||A||^2 + ||B||^2 - 2 * A . B^T
        # This is significantly faster than looping through rows.
        
        # Sum of squares for each row in test and train sets
        dot_test = np.sum(X_test**2, axis=1, keepdims=True)    # (M, 1)
        dot_train = np.sum(self.X_train**2, axis=1)            # (N,)
        
        # Compute the dot product: X_test @ X_train.T (M, N)
        dot_product = np.dot(X_test, self.X_train.T)
        
        # Combine to get the distance matrix (M, N)
        # Broadcasting: (M, 1) + (N,) - (M, N)
        dists = np.sqrt(np.maximum(dot_test + dot_train - 2 * dot_product, 0))
        
        # Find the indices of the k smallest distances for each test point
        # np.argpartition is faster than np.argsort for finding top K
        k_indices = np.argpartition(dists, self.k, axis=1)[:, :self.k]
        
        # Get the labels of the k neighbors
        k_nearest_labels = self.y_train[k_indices]
        
        # Perform majority voting across the rows
        predictions = []
        for labels in k_nearest_labels:
            # Count occurrences of each label
            counts = np.bincount(labels.astype(int))
            predictions.append(np.argmax(counts))
            
        return np.array(predictions)

    def predict_proba(self, X):
        """
        Calculates the probability of the positive class (Churn=1).
        Required for AUC Score calculation.
        """
        X_test = np.array(X)
        
        # Recalculate distances (same optimized logic as predict)
        dot_test = np.sum(X_test**2, axis=1, keepdims=True)
        dot_train = np.sum(self.X_train**2, axis=1)
        dot_product = np.dot(X_test, self.X_train.T)
        dists = np.sqrt(np.maximum(dot_test + dot_train - 2 * dot_product, 0))
        
        k_indices = np.argpartition(dists, self.k, axis=1)[:, :self.k]
        k_nearest_labels = self.y_train[k_indices]
        
        # Probability is the mean of the neighbor labels (since labels are 0 and 1)
        return np.mean(k_nearest_labels, axis=1)

def run_knn(X, y):
    """
    Wrapper function for the Streamlit app integration.
    """
    # Initialize the model with k=5
    model = CustomKNN(k=5)
    
    # Fit the model
    model.fit(X, y)
    
    # Generate predictions and probabilities
    preds = model.predict(X)
    probs = model.predict_proba(X)
    
    return preds, probs