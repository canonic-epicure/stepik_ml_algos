import random
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression, make_classification

X, y = make_classification(n_samples=10, n_features=5, n_informative=2, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]

class MyPCA:
    def __init__(self, n_components=3):
        self.n_components = n_components

    def __str__(self):
        return f'MyPCA class: n_components={ self.n_components }'

    def fit_transform(self, X):
        features = X.to_numpy()
        size = features.shape[ 0 ]

        means = features.mean(axis=0)

        normalized = features - np.repeat([ means ], size, axis=0)

        cov_matrix = np.cov(normalized, rowvar=False)

        w, v = np.linalg.eigh(cov_matrix)

        w_idx = np.argsort(-w, kind="mergesort")

        W = v[ :, w_idx[ :self.n_components ] ]

        return pd.DataFrame(normalized.dot(W))




clf = MyPCA()

clf.fit_transform(X)
