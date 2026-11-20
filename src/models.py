"""Unsupervised detectors. Every model exposes fit(X) and score(X) -> 1-D anomaly score."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM

import os
RNG = int(os.environ.get('SEED', 0))


class RangeEnvelope:
    """Per-sensor min/max envelope learnt on normal data; score = largest excursion."""
    name = "Envelope"
    def fit(self, X):
        self.lo = X.min(0); self.hi = X.max(0); return self
    def score(self, X):
        return np.maximum(np.maximum(self.lo - X, X - self.hi), 0).max(1)


class PCARecon:
    name = "PCA"
    def __init__(self, var=0.95): self.var = var
    def fit(self, X):
        self.m = PCA(n_components=self.var, random_state=RNG).fit(X); return self
    def score(self, X):
        R = self.m.inverse_transform(self.m.transform(X))
        return ((X - R) ** 2).sum(1)


class IForest:
    name = "IForest"
    def fit(self, X):
        self.m = IsolationForest(n_estimators=200, max_samples=1024, random_state=RNG, n_jobs=-1).fit(X); return self
    def score(self, X):
        return -self.m.score_samples(X)


class KNN:
    name = "kNN"
    def __init__(self, k=5, n_ref=20000): self.k, self.n_ref = k, n_ref
    def fit(self, X):
        idx = np.random.default_rng(RNG).choice(len(X), min(self.n_ref, len(X)), replace=False)
        self.m = NearestNeighbors(n_neighbors=self.k, n_jobs=-1).fit(X[idx]); return self
    def score(self, X):
        d, _ = self.m.kneighbors(X); return d.mean(1)


class OCSVM:
    name = "OC-SVM"
    def __init__(self, n_fit=8000): self.n_fit = n_fit
    def fit(self, X):
        idx = np.random.default_rng(RNG).choice(len(X), min(self.n_fit, len(X)), replace=False)
        self.m = OneClassSVM(kernel="rbf", gamma="scale", nu=0.01).fit(X[idx]); return self
    def score(self, X):
        out = np.empty(len(X))
        for i in range(0, len(X), 50000):
            out[i:i + 50000] = -self.m.decision_function(X[i:i + 50000])
        return out


CLASSICAL = [RangeEnvelope, PCARecon, IForest, KNN, OCSVM]
