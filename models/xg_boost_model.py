import numpy as np
from models.decision_tree import CustomDecisionTree

class CustomGradientBoost:
    def __init__(self, n_estimators=5, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.base_pred = None

    def fit(self, X, y):
        self.base_pred = np.mean(y)
        f_m = np.full(len(y), self.base_pred)

        for _ in range(self.n_estimators):
            # Calculate residuals
            residuals = y - f_m
            tree = CustomDecisionTree(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)
            f_m += self.lr * tree.predict(X)

    def predict(self, X):
        f_m = np.full(X.shape[0], self.base_pred)
        for tree in self.trees:
            f_m += self.lr * tree.predict(X)
        return [1 if i > 0.5 else 0 for i in f_m], f_m

def run_xgboost(X, y):
    model = CustomGradientBoost()
    model.fit(X.values, y.values)
    return model.predict(X.values)