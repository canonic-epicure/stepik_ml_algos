import pandas as pd
import numpy as np
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]

class MyLogReg:
    def __init__(self, n_iter, learning_rate):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None

    def __str__(self):
        return f'MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: bool = False):
        eps = 1e-15

        features = np.insert(X.to_numpy(), 0, 1, 1)
        values = y.to_numpy()

        len_feat = features.shape[1]
        len_count = features.shape[0]

        self.weights = np.array([1.0] * len_feat)

        for i in range(self.n_iter):
            prediction = 1 / (1 + np.e ** (-features.dot(self.weights)))

            loss_arr = list(map(
                lambda pred, value: value * np.log(pred + eps) + (1 - value) * np.log(1 - pred + eps),
                prediction,
                values
            ))

            loss = -(np.array(loss_arr).mean())

            if verbose and (i % verbose) == 0:
                print(f'{i} | loss={loss}')

            gradient = 1 / features.shape[0] * features.T.dot(prediction - values)

            self.weights = self.weights - self.learning_rate * gradient

    def get_coef(self):
        return self.weights[1:]

    def predict_proba(self, X: pd.DataFrame):
        features = np.insert(X.to_numpy(), 0, 1, 1)

        return 1 / (1 + np.e ** (-features.dot(self.weights)))

    def predict(self, X: pd.DataFrame):
        return (self.predict_proba(X) >= 0.5) * 1

reg = MyLogReg(n_iter=50, learning_rate=0.1)

reg.fit(X, y, 1)
