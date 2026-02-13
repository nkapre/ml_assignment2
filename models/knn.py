import numpy as np
from collections import Counter

class CustomKNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        print ("Inside fit")
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        print ("Inside predict")
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        print ("Predict Step 1")
        distances = [np.sqrt(np.sum((x - x_train)**2)) for x_train in self.X_train]
        print ("Predict Step 2")
        k_indices = np.argsort(distances)[:self.k]
        print ("Predict Step 3")
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        print ("Predict Step 4")
        most_common = Counter(k_nearest_labels).most_common(1)
        print ("Predict Step 5")

        return most_common[0][0]

def run_knn(X, y):
    print ("Inside run")
    model = CustomKNN(k=5)
    model.fit(X.values, y.values)
    preds = model.predict(X.values)
    return preds, preds