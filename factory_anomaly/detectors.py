"""Two anomaly detectors, both built from scratch in NumPy.

1. IsolationForestDetector - an Isolation Forest (Liu et al., 2008):
   an ensemble of random isolation trees. Anomalies are isolated in
   few splits, so a short average path length means "anomalous".
   Score = 2 ** (-avg_path_length / c(n)), the standard definition.

2. AutoencoderDetector - a small MLP autoencoder
   (input -> 8 -> 2 -> 8 -> input) trained with Adam on normal data
   only. Anomalies reconstruct poorly, so reconstruction error
   (mean squared error per sample) is the anomaly score.

Both are fit on normal data only (semi-supervised) and set their
decision threshold at a percentile of the training scores, so the
caller controls the expected false-alarm rate on normal data.
The default 90th percentile accepts a ~10% false-alarm rate on
normal windows in exchange for catching subtler faults - the right
trade-off for a factory monitor, where a missed fault costs far
more than a checked alarm.
"""

import numpy as np


def _c_factor(n):
    """Average path length of an unsuccessful search in a BST of n."""
    if n <= 1:
        return 1.0
    return 2.0 * (np.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n


class _IsolationTree:
    def __init__(self, rng, max_depth):
        self.rng = rng
        self.max_depth = max_depth

    def fit(self, X, depth=0):
        n = len(X)
        self.size = n
        if n <= 1 or depth >= self.max_depth:
            self.is_leaf = True
            return self
        self.is_leaf = False
        q = self.rng.integers(X.shape[1])          # random feature
        lo, hi = X[:, q].min(), X[:, q].max()
        if lo == hi:
            self.is_leaf = True
            return self
        self.feature = q
        self.split = self.rng.uniform(lo, hi)      # random split value
        left = X[X[:, q] < self.split]
        right = X[X[:, q] >= self.split]
        self.left = _IsolationTree(self.rng, self.max_depth).fit(left, depth + 1)
        self.right = _IsolationTree(self.rng, self.max_depth).fit(right, depth + 1)
        return self

    def path_length(self, x, depth=0):
        if self.is_leaf:
            return depth + _c_factor(self.size)
        if x[self.feature] < self.split:
            return self.left.path_length(x, depth + 1)
        return self.right.path_length(x, depth + 1)


class IsolationForestDetector:
    """Isolation Forest implemented from scratch (no sklearn)."""

    def __init__(self, n_trees=100, sample_size=256, seed=0,
                 threshold_percentile=90.0):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.seed = seed
        self.threshold_percentile = threshold_percentile

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        n = len(X)
        self._sample_size = min(self.sample_size, n)
        max_depth = int(np.ceil(np.log2(self._sample_size)))
        self.trees_ = []
        for _ in range(self.n_trees):
            idx = rng.choice(n, size=self._sample_size, replace=False)
            tree = _IsolationTree(rng, max_depth).fit(X[idx])
            self.trees_.append(tree)
        self._c = _c_factor(self._sample_size)
        train_scores = self.score_samples(X)
        self.threshold_ = float(np.percentile(train_scores,
                                              self.threshold_percentile))
        return self

    def score_samples(self, X):
        """Anomaly score in [0, 1]; higher = more anomalous."""
        paths = np.array([
            [t.path_length(x) for t in self.trees_] for x in X
        ])
        avg_path = paths.mean(axis=1)
        return 2.0 ** (-avg_path / self._c)

    def predict(self, X):
        return (self.score_samples(X) >= self.threshold_).astype(int)


def _relu(z):
    return np.maximum(0.0, z)


def _relu_grad(z):
    return (z > 0).astype(float)


class AutoencoderDetector:
    """Small NumPy autoencoder: d -> 8 -> 2 -> 8 -> d, trained by Adam.

    Trained on normal data only; the anomaly score is the per-sample
    mean squared reconstruction error.
    """

    def __init__(self, hidden=8, bottleneck=2, epochs=300, lr=0.01,
                 batch_size=64, seed=0, threshold_percentile=90.0):
        self.hidden = hidden
        self.bottleneck = bottleneck
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.threshold_percentile = threshold_percentile

    def _init_params(self, d):
        rng = np.random.default_rng(self.seed)

        def w(fan_in, fan_out):
            return rng.normal(0, np.sqrt(2.0 / fan_in), (fan_in, fan_out))

        return {
            "W1": w(d, self.hidden), "b1": np.zeros(self.hidden),
            "W2": w(self.hidden, self.bottleneck), "b2": np.zeros(self.bottleneck),
            "W3": w(self.bottleneck, self.hidden), "b3": np.zeros(self.hidden),
            "W4": w(self.hidden, d), "b4": np.zeros(d),
        }

    @staticmethod
    def _forward(params, X):
        z1 = X @ params["W1"] + params["b1"]
        a1 = _relu(z1)
        z2 = a1 @ params["W2"] + params["b2"]
        a2 = _relu(z2)
        z3 = a2 @ params["W3"] + params["b3"]
        a3 = _relu(z3)
        out = a3 @ params["W4"] + params["b4"]
        cache = (X, z1, a1, z2, a2, z3, a3, out)
        return out, cache

    @staticmethod
    def _backward(params, cache):
        X, z1, a1, z2, a2, z3, a3, out = cache
        n = len(X)
        d_out = 2.0 * (out - X) / (n * X.shape[1])   # d MSE / d out
        grads = {}
        grads["W4"] = a3.T @ d_out
        grads["b4"] = d_out.sum(axis=0)
        d_a3 = d_out @ params["W4"].T
        d_z3 = d_a3 * _relu_grad(z3)
        grads["W3"] = a2.T @ d_z3
        grads["b3"] = d_z3.sum(axis=0)
        d_a2 = d_z3 @ params["W3"].T
        d_z2 = d_a2 * _relu_grad(z2)
        grads["W2"] = a1.T @ d_z2
        grads["b2"] = d_z2.sum(axis=0)
        d_a1 = d_z2 @ params["W2"].T
        d_z1 = d_a1 * _relu_grad(z1)
        grads["W1"] = X.T @ d_z1
        grads["b1"] = d_z1.sum(axis=0)
        return grads

    def fit(self, X):
        rng = np.random.default_rng(self.seed + 1)
        self.params_ = self._init_params(X.shape[1])
        # Adam state
        m = {k: np.zeros_like(v) for k, v in self.params_.items()}
        v = {k: np.zeros_like(v) for k, v in self.params_.items()}
        step = 0
        n = len(X)
        for _ in range(self.epochs):
            perm = rng.permutation(n)
            for i in range(0, n, self.batch_size):
                batch = X[perm[i:i + self.batch_size]]
                _, cache = self._forward(self.params_, batch)
                grads = self._backward(self.params_, cache)
                step += 1
                for k in self.params_:
                    m[k] = 0.9 * m[k] + 0.1 * grads[k]
                    v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
                    m_hat = m[k] / (1 - 0.9 ** step)
                    v_hat = v[k] / (1 - 0.999 ** step)
                    self.params_[k] -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        train_scores = self.score_samples(X)
        self.threshold_ = float(np.percentile(train_scores,
                                              self.threshold_percentile))
        return self

    def score_samples(self, X):
        out, _ = self._forward(self.params_, X)
        return ((out - X) ** 2).mean(axis=1)

    def predict(self, X):
        return (self.score_samples(X) >= self.threshold_).astype(int)
