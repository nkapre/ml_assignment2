import numpy as np
from collections import Counter

class CustomKNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y


    def predict(self, X):
        X = np.array(X)
        # 1. Vectorized Euclidean Distance: sqrt(sum((X - X_train)^2))
        # Using broadcasting to avoid the 'for x_train in self.X_train' loop
        # (A-B)^2 = A^2 + B^2 - 2AB
        dists = np.sqrt(
            np.sum(X**2, axis=1)[:, np.newaxis] + 
            np.sum(self.X_train**2, axis=1) - 
            2 * np.dot(X, self.X_train.T)
        )

        # 2. Get the indices of the k smallest distances
        k_indices = np.argsort(dists, axis=1)[:, :self.k]

        # 3. Vote for labels
        predictions = []
        for indices in k_indices:
            k_nearest_labels = self.y_train[indices]
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])
            
        return np.array(predictions)

    def _predict(self, x):
        distances = [np.sqrt(np.sum((x - x_train)**2)) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

def run_knn(X, y):
    model = CustomKNN(k=5)
    model.fit(X.values, y.values)
    preds = model.predict(X.values)
    return preds, preds