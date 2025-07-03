import random
import pandas as pd
import numpy as np

from sklearn.datasets import make_regression, make_classification
# X, y = make_regression(n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42)
# X = pd.DataFrame(X)
# y = pd.Series(y)
# X.columns = [f'col_{col}' for col in X.columns]

# X, y = make_classification(n_samples=10, n_features=5, n_informative=2, random_state=42)
# X = pd.DataFrame(X)
# y = pd.Series(y)
# X.columns = [f'col_{col}' for col in X.columns]

from temp import X

dtype = [ ('from', 'i4'), ('to', 'i4'), ('distance', 'f4') ]

class Point:
    def __init__(self, point, index, distances_to_others):
        self.point = point
        self.index = index

        self.cluster = None
        self.kind = None
        self.distances_to_others = distances_to_others


class MyDBSCAN:
    def __init__(self, eps=3, min_samples=3, metric='euclidean'):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

    def __str__(self):
        return f'MyDBSCAN class: eps={ self.eps }, min_samples={ self.min_samples }'

    def get_distance(self, point1, point2):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((point1 - point2) ** 2, axis=0))
        elif self.metric == 'chebyshev':
            return np.abs(point1 - point2).max(axis=0)
        elif self.metric == 'manhattan':
            return np.abs(point1 - point2).sum(axis=0)
        elif self.metric == 'cosine':
            scalar_prod = (point1 * point2).sum(axis=0)
            distance_prod = np.sqrt((point1 ** 2).sum(axis=0)) * np.sqrt((point2 ** 2).sum(axis=0))

            return 1 - scalar_prod / distance_prod
        else:
            # should not happen
            raise RuntimeError(f"Unknown metric={self.metric}")

    def get_distances_to_others(self, features, idx, distances):
        index = np.stack(np.indices(distances.shape), axis=-1)

        from_to_idx = index[ idx, idx + 1: ]
        from_to_distances = distances[ from_to_idx[:, 0], from_to_idx[:, 1] ]

        to_from_idx = index[ :idx, idx ]
        to_from_distances = distances[ to_from_idx[:, 0], to_from_idx[:, 1] ]

        dist = np.concatenate(
            [
                np.array(
                    list(zip(
                        from_to_idx[:, 0],
                        from_to_idx[:, 1],
                        from_to_distances
                    )),
                    dtype=dtype
                ),
                np.array(
                    list(zip(
                        to_from_idx[:, 1],
                        to_from_idx[:, 0],
                        to_from_distances
                    )),
                    dtype=dtype
                )
            ],
            axis=0
        )

        dist.sort(order='distance')

        return dist[ dist['distance'] < self.eps ]


    def fit_predict(self, X):
        features = X.to_numpy()
        size = features.shape[ 0 ]

        distances = np.zeros(shape=(size, size))
        for i in range(size):
            for j in range(i + 1, size):
                distances[ i, j ] = self.get_distance(features[ i ], features[ j ])

        els = np.array(
            [
                Point(
                    point = features[ i ], index = i, distances_to_others = self.get_distances_to_others(features, i, distances)
                )
                    for i in range(size)
            ],
            dtype='object'
        )

        stack = list(els[::-1])

        cluster_id = 0

        while len(stack) > 0:
            point = stack.pop(-1)

            if point.kind is None:
                if point.distances_to_others.shape[0] >= self.min_samples:
                    point.kind = 'core'
                    neighbors = els[ point.distances_to_others['to'] ]

                    cluster_id += 1
                    point.cluster = cluster_id

                    for neighbor in neighbors:
                        if neighbor.kind != 'core':
                            neighbor.kind = 'border'
                            stack.append(neighbor)
                else:
                    point.kind = 'outlier'

            elif point.kind == 'border':
                if point.cluster is None:
                    if point.distances_to_others.shape[0] >= self.min_samples:
                        point.kind = 'core'
                        neighbors = els[ point.distances_to_others['to'] ]

                        point.cluster = cluster_id

                        for neighbor in neighbors:
                            if neighbor.kind != 'core' and neighbor.kind != 'border':
                                neighbor.kind = 'border'
                                stack.append(neighbor)
                    else:
                        point.cluster = cluster_id

            elif point.kind == 'core':
                pass

            elif point.kind == 'outlier':
                pass
            else:
                raise "Unknown point kind"

        cluster_id += 1

        return [ point.cluster if point.cluster is not None else cluster_id for point in els ]

clf = MyDBSCAN(eps=2, min_samples=3)

pred = clf.fit_predict(X)

values, counts = np.unique(pred, return_counts=True)

counts.sort()

print(counts)