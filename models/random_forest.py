from models.decision_tree import CustomDecisionTree
import numpy as np

class CustomRandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_trees):
            tree = CustomDecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            # Bootstrap sample
            idx = np.random.choice(len(X), len(X), replace=True)
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # Majority vote
        return np.round(np.mean(tree_preds, axis=0)).astype(int)

def run_random_forest(X, y):
    model = CustomRandomForest(n_trees=5)
    model.fit(X.values, y.values)
    preds = model.predict(X.values)
    return preds, preds