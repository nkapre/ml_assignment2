import numpy as np
from collections import Counter
from scipy.spatial import KDTree
import streamlit as st

class CustomKNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        ####self.X_train = X
        ####self.y_train = y
        self.tree = KDTree(X)
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        # X: (m_samples, n_features), self.X_train: (n_samples, n_features)
        # Compute all distances at once using broadcasting
        # (a-b)^2 = a^2 - 2ab + b^2
        dists = np.sqrt(np.sum(X**2, axis=1)[:, np.newaxis] + 
                        np.sum(self.X_train**2, axis=1) - 
                        2 * np.dot(X, self.X_train.T))
        
        # Get indices of k smallest distances for each row
        ####k_indices = np.argsort(dists, axis=1)[:, :self.k]
        _, k_indices = self.tree.query(X, k=self.k)
        
        # Map indices to labels
        k_nearest_labels = self.y_train[k_indices]
        
        # Return most common label per row
        return np.array([Counter(row).most_common(1)[0][0] for row in k_nearest_labels])

@st.cache_data
def run_knn(X, y):
    model = CustomKNN(k=5)
    model.fit(X.values, y.values)
    preds = model.predict(X.values)
    return preds, preds