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


def get_distance_square(point1, point2):
    return ((point1 - point2) ** 2).sum(axis=1)

def get_distance(point1, point2):
    return np.sqrt(((point1 - point2) ** 2).sum(axis=1))

class MyKMeansSingle:
    def __init__(self, n_clusters=3, max_iter=10):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.cluster_origins = None
        self.cluster_labels = None
        self.features = None

    def fit(self, X):
        features = self.features = X.to_numpy()

        f_min = features.min(axis=0)
        f_max = features.max(axis=0)

        cluster_origins = np.array([
            np.random.uniform(f_min, f_max) for _ in range(self.n_clusters)
        ])

        cluster_labels = None

        i = 0

        while True:
            distances = np.array([
                get_distance(
                    np.repeat([ cluster_origin ], X.shape[ 0 ], axis=0),
                    features
                )
                    for cluster_origin in cluster_origins
            ])

            distance_idx = distances.argsort(axis=0, kind="mergesort")

            cluster_labels = distance_idx[ 0 ]

            distance_idx_with_sample_idx = np.array(list(enumerate(distance_idx[ 0 ])))

            def get_new_cluster_center(c):
                samples_of_cluster = features[
                    distance_idx_with_sample_idx[
                        distance_idx_with_sample_idx[:, 1] == c,
                        0
                    ]
                ]

                if len(samples_of_cluster) == 0:
                    return cluster_origins[ c ]
                else:
                    return samples_of_cluster.mean(axis=0)


            new_centers = np.array([ get_new_cluster_center(c) for c in range(self.n_clusters) ])

            i += 1


            nan_idx = np.isnan(new_centers)
            new_centers[nan_idx] = cluster_origins[nan_idx]

            if i >= self.max_iter or np.array_equal(new_centers, cluster_origins):
                break

            cluster_origins = new_centers

        self.cluster_origins = cluster_origins
        self.cluster_labels = cluster_labels

    def get_wcss(self):
        return get_distance_square(self.features, self.cluster_origins[ self.cluster_labels ]).sum()


class MyKMeans:
    def __init__(self, n_clusters=3, max_iter=10, n_init=3, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

        self.k_mean = None
        self.inertia_ = None
        self.cluster_centers_ = None

    def __str__(self):
        return f'MyKMeans class: n_clusters={ self.n_clusters }, max_iter={ self.max_iter }, n_init={ self.n_init }, random_state={ self.random_state }'

    def fit(self, X):
        np.random.seed(self.random_state)

        k_means = [ MyKMeansSingle(n_clusters=self.n_clusters, max_iter=self.max_iter) for _ in range(self.n_init) ]

        for k_mean in k_means:
            k_mean.fit(X)

        wcss_s = np.array([ k_mean.get_wcss() for k_mean in k_means ])

        wcss_s_idx = wcss_s.argsort(kind="mergesort")

        self.k_mean = k_means[ wcss_s_idx[ 0 ] ]
        self.inertia_ = wcss_s[ wcss_s_idx[ 0 ] ]
        self.cluster_centers_ = self.k_mean.cluster_origins

    def predict(self, X):
        features = X.to_numpy()

        distances = np.array([
            get_distance(
                np.repeat([ cluster_origin ], X.shape[ 0 ], axis=0),
                features
            )
                for cluster_origin in self.cluster_centers_
        ])

        distance_idx = distances.argsort(axis=0, kind="mergesort")

        return distance_idx[ 0 ]


clf = MyKMeans()

clf.fit(X)
