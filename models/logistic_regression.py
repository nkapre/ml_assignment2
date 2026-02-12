import numpy as np

class CustomLogisticRegression:
    def __init__ (self, lr=0.01, iterations=1000):
        self.lr = lr
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        z = np.array(z, dtype=float)
        return 1 / (1 + np.exp(-z))


    def fit(self, X, y):
        # 1. Force conversion to NumPy float64 arrays to avoid 'O' (Object) dtypes
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64).reshape(-1, 1) # Ensure y is a column vector

        n_samples, n_features = X.shape

        # 2. Initialize weights explicitly as float64
        self.weights = np.zeros((n_features, 1), dtype=np.float64)
        self.bias = 0.0

        for _ in range(self.iterations):
            model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(model)

    def predict(self, X):
        return [1 if i > 0.5 else 0 for i in self.predict_proba(X)]

def run_logistic_regression(X, y):
    model = CustomLogisticRegression()
    model.fit(X.values, y.values)
    return model.predict(X.values), model.predict_proba(X.values)