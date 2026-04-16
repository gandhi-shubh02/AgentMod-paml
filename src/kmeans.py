from typing import Dict, Iterable

import numpy as np


class KMeans:
    def __init__(self, K=4, max_iter=100, tol=1e-4, seed=42):
        self.K = K
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None

    def _pairwise_squared_distances(self, X, Y):
        x2 = np.sum(X * X, axis=1, keepdims=True)
        y2 = np.sum(Y * Y, axis=1, keepdims=True).T
        return np.maximum(x2 + y2 - 2.0 * (X @ Y.T), 0.0)

    def _init_centroids_plusplus(self, X):
        n_samples = X.shape[0]
        centroids = []
        first_idx = self.rng.integers(0, n_samples)
        centroids.append(X[first_idx].copy())

        for _ in range(1, self.K):
            C = np.vstack(centroids)
            d2 = self._pairwise_squared_distances(X, C)
            min_d2 = np.min(d2, axis=1)
            probs = min_d2 / np.sum(min_d2)
            next_idx = self.rng.choice(n_samples, p=probs)
            centroids.append(X[next_idx].copy())

        return np.vstack(centroids)

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.centroids = self._init_centroids_plusplus(X)

        for _ in range(self.max_iter):
            d2 = self._pairwise_squared_distances(X, self.centroids)
            labels = np.argmin(d2, axis=1)

            new_centroids = np.zeros_like(self.centroids)
            for k in range(self.K):
                members = X[labels == k]
                if members.shape[0] == 0:
                    new_centroids[k] = X[self.rng.integers(0, X.shape[0])]
                else:
                    new_centroids[k] = members.mean(axis=0)

            shift = np.linalg.norm(new_centroids - self.centroids, axis=1).max()
            self.centroids = new_centroids
            if shift < self.tol:
                break

        final_d2 = self._pairwise_squared_distances(X, self.centroids)
        self.labels_ = np.argmin(final_d2, axis=1)
        self.inertia_ = float(np.sum(np.min(final_d2, axis=1)))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        d2 = self._pairwise_squared_distances(X, self.centroids)
        return np.argmin(d2, axis=1)

    def silhouette_score(self, X):
        X = np.asarray(X, dtype=np.float64)
        labels = self.predict(X) if self.labels_ is None or len(self.labels_) != len(X) else self.labels_
        n = X.shape[0]
        if n < 2:
            return 0.0

        unique_labels = np.unique(labels)
        if unique_labels.size < 2:
            return 0.0

        dists = np.sqrt(self._pairwise_squared_distances(X, X))
        s_vals = np.zeros(n, dtype=np.float64)

        for i in range(n):
            own = labels[i]
            own_mask = labels == own
            own_count = np.sum(own_mask)
            if own_count <= 1:
                a_i = 0.0
            else:
                a_i = float(np.sum(dists[i, own_mask]) / (own_count - 1))

            b_i = np.inf
            for other in unique_labels:
                if other == own:
                    continue
                other_mask = labels == other
                if np.any(other_mask):
                    b_i = min(b_i, float(np.mean(dists[i, other_mask])))

            denom = max(a_i, b_i)
            s_vals[i] = 0.0 if denom == 0.0 else (b_i - a_i) / denom

        return float(np.mean(s_vals))

    def save(self, path):
        np.save(path, self.centroids)

    def load(self, path):
        self.centroids = np.load(path)


def elbow_analysis(X, k_range=range(2, 9), seed=42):
    results: Dict[int, Dict[str, float]] = {}
    for k in k_range:
        km = KMeans(K=k, max_iter=100, tol=1e-4, seed=seed)
        km.fit(X)
        sil = km.silhouette_score(X)
        results[int(k)] = {"inertia": float(km.inertia_), "silhouette": float(sil)}
    return results

