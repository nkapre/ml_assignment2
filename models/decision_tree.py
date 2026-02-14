import numpy as np
from collections import Counter

class Node:
    __slots__ = ['feature', 'threshold', 'left', 'right', 'value']
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    

class CustomDecisionTree:
    #def __init__(self, max_depth=10, min_samples_split=2):
    def __init__(self, max_depth=5, min_samples_split=10, n_bins=20):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.n_bins = n_bins



    def fit(self, X, y):
        y = y.astype(np.float64)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        #if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
        if (depth >= self.max_depth or n_samples < self.min_samples_split or np.var(y) < 1e-7):
            #leaf_value = Counter(y).most_common(1)[0][0]
            leaf_value = np.mean(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, n_feats, replace=False)
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)

        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]

            # FIX 3: Quantile Binning - Only check 20 points instead of every unique value
            if len(X_column) > self.n_bins:
                thresholds = np.percentile(X_column, np.linspace(0, 100, self.n_bins))
            else:
                thresholds = np.unique(X_column)

            for thresh in thresholds:
                #gain = self._information_gain(y, X_column, thresh)
                gain = self._variance_reduction(y, X_column, thresh)
                if gain > best_gain:
                    best_gain, split_idx, split_thresh = gain, feat_idx, thresh

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, thresh):
        parent_entropy = self._entropy(y)
        left_idxs, right_idxs = self._split(X_column, thresh)
        if len(left_idxs) == 0 or len(right_idxs) == 0: return 0
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        return parent_entropy - child_entropy

    def _variance_reduction(self, y, X_column, thresh):
    #    parent_var = np.var(y)
    #    left_idxs, right_idxs = self._split(X_column, thresh)
    #    if len(left_idxs) == 0 or len(right_idxs) == 0: return 0
    #    
    #    n = len(y)
    #    n_l, n_r = len(left_idxs), len(right_idxs)
    #    var_l, var_r = np.var(y[left_idxs]), np.var(y[right_idxs])
    #    
    #    child_var = (n_l / n) * var_l + (n_r / n) * var_r
    #    return parent_var - child_var


        #Boolean masking is significantly faster than np.argwhere
        left_mask = X_column <= thresh
        right_mask = ~left_mask 
        
        if not np.any(left_mask) or not np.any(right_mask): 
            return 0
        
        n = len(y)
        n_l, n_r = np.sum(left_mask), np.sum(right_mask)
        # FIX 5: Vectorized variance reduction calculation
        return np.var(y) - ((n_l/n) * np.var(y[left_mask]) + (n_r/n) * np.var(y[right_mask]))


    #def _entropy(self, y):
    #    hist = np.bincount(y)
    #    ps = hist / len(y)
    #    return -np.sum([p * np.log(p) for p in ps if p > 0])

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    #def predict(self, X):
    #    return np.array([self._traverse_tree(x, self.root) for x in X])

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        self._predict_vectorized(X, np.arange(X.shape[0]), self.root, predictions)
        return predictions

    def _predict_vectorized(self, X, indices, node, predictions):
        if node.value is not None:
            predictions[indices] = node.value
            return

        feat_values = X[indices, node.feature]
        left_mask = feat_values <= node.threshold
        
        if np.any(left_mask):
            self._predict_vectorized(X, indices[left_mask], node.left, predictions)
        if np.any(~left_mask):
            self._predict_vectorized(X, indices[~left_mask], node.right, predictions)


    def _traverse_tree(self, x, node):
        if node.value is not None: return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

def run_decision_tree(X, y):
    # This remains a classifier for the "Decision Tree" menu option in app.py
    # But note: The internal logic above now supports XGBoost's floats
    model = CustomDecisionTree()
    model.fit(X.values, y.values)
    preds = model.predict(X.values)
    # Binary threshold for the standalone tree option
    binary_preds = np.array([1 if i > 0.5 else 0 for i in preds])
    return binary_preds, preds
    