import random
import pandas as pd
import numpy as np

from sklearn.datasets import make_regression, make_classification
# X, y = make_regression(n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42)
# X = pd.DataFrame(X)
# y = pd.Series(y)
# X.columns = [f'col_{col}' for col in X.columns]

X, y = make_classification(n_samples=10, n_features=5, n_informative=2, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]


class Point:
    def __init__(self, point, index):
        self.point = point
        self.index = index

    def get_center(self):
        return self.point

    def get_points(self):
        return [ self ]

    def gen_points(self):
        yield self


class Cluster:
    def __init__(self, elements = []):
        self.elements = elements

        self.center = None
        self.points = None

    def get_center(self):
        if (self.center is not None):
            return self.center

        self.center = self.build_center()

        return self.center

    def get_points(self):
        if (self.points is not None):
            return self.points

        self.points = self.build_points()

        return self.points

    def build_center(self):
        return np.array([ p.point for p in self.get_points() ]).mean(axis=0)

    def build_points(self):
        return list(self.gen_points())

    def gen_points(self):
        for element in self.elements:
            yield from element.gen_points()


class MyAgglomerative:
    def __init__(self, n_clusters=3, metric='euclidean'):
        self.n_clusters = n_clusters
        self.metric = metric

    def __str__(self):
        return f'MyAgglomerative class: n_clusters={ self.n_clusters }'

    def get_distance(self, point1, point2):
        if self.metric == 'euclidean':
            return np.sqrt(
                np.sum((point1.get_center() - point2.get_center()) ** 2, axis=0)
            )
        elif self.metric == 'chebyshev':
            return np.abs(point1.get_center() - point2.get_center()).max(axis=0)
        elif self.metric == 'manhattan':
            return np.abs(point1.get_center() - point2.get_center()).sum(axis=0)
        elif self.metric == 'cosine':
            scalar_prod = (point1.get_center() * point2.get_center()).sum(axis=0)
            distance_prod = np.sqrt((point1.get_center() ** 2).sum(axis=0)) * np.sqrt((point2.get_center() ** 2).sum(axis=0))

            return 1 - scalar_prod / distance_prod
        else:
            # should not happen
            raise RuntimeError(f"Unknown metric={self.metric}")

    def fit_predict(self, X):
        features = X.to_numpy()

        els = [ Point(point = features[ i ], index = i) for i in range(features.shape[ 0 ]) ]

        while len(els) > 2 and len(els) > self.n_clusters:
            size = len(els)

            distances = np.array([
                [ i, j, self.get_distance(els[ i ], els[ j ]) ]
                    for i in range(size)
                        for j in range(i + 1, size)
            ])

            distances_idx = distances[ :, 2 ].argmin(axis=0)

            closest = distances[ distances_idx ]

            combined = Cluster(elements = [
                els[ int(closest[ 0 ]) ], els[ int(closest[ 1 ]) ]
            ])

            els[ int(closest[ 0 ]) ] = combined
            els[ int(closest[ 1 ]) : int(closest[ 1 ]) + 1 ] = []

        indicies = [ None ] * features.shape[ 0 ]

        for idx, el in enumerate(els):
            for point in el.get_points():
                indicies[ point.index ] = idx

        return indicies


clf = MyAgglomerative()

clf.fit_predict(X)
