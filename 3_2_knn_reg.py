import random
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression

from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=14, n_informative=10, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]

class MyKNNReg:
    def __init__(self, k=3, metric='euclidean', weight='uniform'):
        self.k = k
        self.train_size = 0
        self.features = None
        self.values = None
        self.metric = metric
        self.weight = weight

    def __str__(self):
        return f'MyKNNReg class: k={self.k}'

    def fit(self, X, y):
        self.features = X.to_numpy()
        self.values = y.to_numpy()
        self.train_size = self.features.shape

    def get_distance(self, point1, point2):
        if (self.metric == 'euclidean'):
            return np.sqrt(np.sum((point1 - point2) ** 2, axis=1))
        elif self.metric == 'chebyshev':
            return np.abs(point1 - point2).max(axis=1)
        elif self.metric == 'manhattan':
            return np.abs(point1 - point2).sum(axis=1)
        elif self.metric == 'cosine':
            scalar_prod = (point1 * point2).sum(axis=1)
            distance_prod = np.sqrt((point1 ** 2).sum(axis=1)) * np.sqrt((point2 ** 2).sum(axis=1))

            return 1 - scalar_prod / distance_prod
        else:
            # should not happen
            raise RuntimeError(f"Unknown metric={self.metric}")

    def predict(self, X):
        features = X.to_numpy()

        res = []

        for xi in features:
            xi_repeated = np.repeat([ xi ], self.features.shape[-2], axis=0)
            distances = self.get_distance(xi_repeated, self.features)

            sorted_idx = np.argsort(distances, axis=0)
            targets = self.values[ sorted_idx[ 0:self.k ] ]
            sorted_distances = distances[ sorted_idx[ 0:self.k ] ]

            if self.weight == 'uniform':
                res.append(targets.mean(axis=0))
            elif self.weight == 'rank':
                ranks = np.array(range(self.k + 1)[1:])

                res.append((targets * (1 / ranks)).sum(axis=0) / ((1 / ranks).sum()))

            elif self.weight == 'distance':
                res.append((targets * (1 / sorted_distances)).sum(axis=0) / (1 / sorted_distances).sum())
            else:
                raise RuntimeError(f"Unknown weight={self.weight}")


        return np.array(res)


reg = MyKNNReg(k=3, weight='rank')

reg.fit(X, y)

reg.predict(X)