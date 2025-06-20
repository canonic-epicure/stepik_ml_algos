import random
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]


class MySVM:
    def __init__(self, n_iter=10, learning_rate=0.001, C=1, sgd_sample=None, random_state = 42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.b = None
        self.C = C
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def __str__(self):
        return f'MySVM class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'

    def get_learning_selection_idx(self, features):
        if self.sgd_sample == None:
            return None
        else:
            count = self.sgd_sample if isinstance(self.sgd_sample, int) else round(self.sgd_sample * features.shape[0])
            return random.sample(range(features.shape[0]), count)

    def fit(self, X, y, verbose=False):
        random.seed(self.random_state)

        features = X.to_numpy()
        # translate to [ -1, 1 ]
        values = y.to_numpy() * 2 - 1

        self.weights = np.ones(features.shape[-1])
        self.b = 1.0

        for i in range(self.n_iter):
            learning_idx = self.get_learning_selection_idx(features)
            learning_features = features if learning_idx == None else features[learning_idx]
            learning_values = values if learning_idx == None else values[learning_idx]

            for k in range(learning_features.shape[-2]):
                cond = (learning_values[ k ] * (self.weights.T @ learning_features[ k ] + self.b)) >= 1

                delta_weights = None
                delta_b = 0

                if cond:
                    delta_weights = 2 * self.weights
                    delta_b = 0
                else:
                    delta_weights = 2 * self.weights - self.C * learning_values[ k ] * learning_features[ k ]
                    delta_b = -self.C * learning_values[ k ]

                self.weights = self.weights - self.learning_rate * delta_weights
                self.b = self.b - self.learning_rate * delta_b

            if verbose and (i % verbose) == 0:
                loss = (self.weights ** 2).sum() + self.C * np.maximum(0, 1 - values @ (features @ self.weights + self.b)).mean()

    def get_coef(self):
        return (self.weights, self.b)

    def predict(self, X):
        features = X.to_numpy()

        return np.intc((np.sign(features @ self.weights + self.b) + 1) / 2)



reg = MySVM(n_iter=50, learning_rate=0.1)

reg.fit(X, y, 1)
