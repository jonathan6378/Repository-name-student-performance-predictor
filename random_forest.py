"""
Random Forest Regressor implemented from scratch.

Uses only Python standard library and NumPy.
No sklearn tree/forest implementation is used.
"""

import numpy as np


class DecisionTreeRegressorScratch:
    def __init__(self, max_depth=8, min_samples_split=4,
                 max_features="sqrt", random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.rng = np.random.default_rng(random_state)
        self.root = None
        self.n_features_ = None
        self.feature_importances_ = None

    def _mse(self, y):
        if len(y) == 0:
            return 0.0
        mean = np.mean(y)
        return np.mean((y - mean) ** 2)

    def _candidate_features(self):
        if self.max_features == "sqrt":
            k = max(1, int(np.sqrt(self.n_features_)))
        elif self.max_features == "log2":
            k = max(1, int(np.log2(self.n_features_)))
        elif isinstance(self.max_features, int):
            k = min(self.n_features_, self.max_features)
        else:
            k = self.n_features_
        return self.rng.choice(self.n_features_, size=k, replace=False)

    def _best_split(self, X, y):
        parent_error = self._mse(y) * len(y)
        best_gain = 0.0
        best_feature = None
        best_threshold = None

        for feature in self._candidate_features():
            values = np.unique(X[:, feature])
            if len(values) <= 1:
                continue

            thresholds = (values[:-1] + values[1:]) / 2.0

            for threshold in thresholds:
                left = X[:, feature] <= threshold
                right = ~left
                if left.sum() < 1 or right.sum() < 1:
                    continue

                left_y, right_y = y[left], y[right]
                weighted_error = (
                    len(left_y) * self._mse(left_y)
                    + len(right_y) * self._mse(right_y)
                )
                gain = parent_error - weighted_error

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build(self, X, y, depth):
        node = {"value": float(np.mean(y))}

        if (depth >= self.max_depth or
            len(y) < self.min_samples_split or
            np.allclose(y, y[0])):
            return node

        feature, threshold, gain = self._best_split(X, y)

        if feature is None:
            return node

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return node

        self.feature_importances_[feature] += gain

        node.update({
            "feature": int(feature),
            "threshold": float(threshold),
            "left": self._build(X[left_mask], y[left_mask], depth + 1),
            "right": self._build(X[right_mask], y[right_mask], depth + 1),
        })
        return node

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_ = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features_)
        self.root = self._build(X, y, 0)

        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
        return self

    def _predict_one(self, row, node):
        if "feature" not in node:
            return node["value"]
        if row[node["feature"]] <= node["threshold"]:
            return self._predict_one(row, node["left"])
        return self._predict_one(row, node["right"])

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(row, self.root) for row in X])


class RandomForestRegressorScratch:
    def __init__(self, n_estimators=30, max_depth=8,
                 min_samples_split=4, max_features="sqrt",
                 random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.feature_importances_ = None

    def _bootstrap(self, X, y, rng):
        n = len(X)
        indices = rng.integers(0, n, size=n)
        return X[indices], y[indices]

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.random_state)
        self.trees = []
        importances = np.zeros(X.shape[1])

        for i in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap(X, y, rng)

            tree = DecisionTreeRegressorScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=self.random_state + i + 1
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            importances += tree.feature_importances_

        self.feature_importances_ = importances / self.n_estimators
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
        return self

    def predict(self, X):
        if not self.trees:
            raise ValueError("Model has not been trained.")
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(predictions, axis=0)
