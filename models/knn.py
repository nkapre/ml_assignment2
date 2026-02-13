import numpy as np
from collections import Counter

class CustomKNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        print ("Inside fit")
        #self.X_train = X
        #self.y_train = y
        self.X_train = np.array(X, dtype=np.float64)
        self.y_train = np.array(y)

    def predict(self, X):
        # Ensure the input to be predicted is also float64
        X = np.array(X, dtype=np.float64)
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)


    #def predict(self, X):
    #    print ("Inside predict")
    #    predictions = [self._predict(x) for x in X]
    #    return np.array(predictions)

    #def _predict(self, x):
    #    print ("Predict Step 1")
    #    distances = [np.sqrt(np.sum((x - x_train)**2)) for x_train in self.X_train]
    #    print ("Predict Step 2")
    #    k_indices = np.argsort(distances)[:self.k]
    #    print ("Predict Step 3")
    #    k_nearest_labels = [self.y_train[i] for i in k_indices]
    #    print ("Predict Step 4")
    #    most_common = Counter(k_nearest_labels).most_common(1)
    #    print ("Predict Step 5")

    #    return most_common[0][0]

    def _predict(self, x):
        # 1. Vectorized Euclidean Distance
        # This replaces the list comprehension loop
        print ("Predict step 1")
        distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
        
        # 2. Use argpartition instead of argsort
        # argpartition is O(N) while argsort is O(N log N)
        # It finds the k smallest elements without sorting the whole array
        print ("Predict step 2")
        k_indices = np.argpartition(distances, self.k)[:self.k]
        
        # 3. Vectorized indexing
        print ("Predict step 3")
        ##k_nearest_labels = self.y_train[k_indices]
        k_nearest_labels = self.y_train[k_indices].flatten()
        
        # 4. Majority vote
        print ("Predict step 4")
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

def run_knn(X, y):
    print ("Inside run")
    model = CustomKNN(k=5)
    model.fit(X.values, y.values)
    preds = model.predict(X.values)
    return preds, preds