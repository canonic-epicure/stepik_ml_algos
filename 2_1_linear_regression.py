import random
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]

class MyLineReg:
    def __init__(self, n_iter, learning_rate, metric=None, reg=None, l1_coef=0.0, l2_coef=0.0, sgd_sample=None, random_state = 42):
        self.metric = metric
        self.weights = None
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.mse = None
        self.mae = None
        self.rmse = None
        self.mape = None
        self.r2 = None
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def __str__(self):
        return f'MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'

    def get_learning_rate(self, iteration: int) -> float:
        if callable(self.learning_rate):
            return self.learning_rate(iteration)
        else:
            return self.learning_rate

    def learning_selection(self, X: np.array, sgd_sample):
        if sgd_sample == None:
            return None

        count = sgd_sample if isinstance(sgd_sample, int) else round(sgd_sample * X.shape[0])

        return random.sample(range(X.shape[0]), count)

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: bool = False):
        random.seed(self.random_state)

        X.insert(0, '__x0__', 1.0)
        len_feat = X.shape[1]
        len_count = X.shape[0]

        weights = np.array([1.0] * len_feat)

        features = X.to_numpy()
        values = y.to_numpy()

        loss = lambda: ((features.dot(weights) - values) ** 2).mean()

        for i in range(self.n_iter):
            learning_selection = self.learning_selection(features, self.sgd_sample)
            learning_features = features if learning_selection == None else features[learning_selection]
            learning_values = values if learning_selection == None else values[learning_selection]

            prediction = learning_features.dot(weights)

            gradient = 2 / learning_features.shape[0] * learning_features.T.dot(prediction - learning_values)

            if self.reg == 'l1':
                gradient += self.l1_coef * np.sign(weights)
            elif self.reg == 'l2':
                gradient += 2 * self.l2_coef * weights
            elif self.reg == 'elasticnet':
                gradient += self.l1_coef * np.sign(weights) + 2 * self.l2_coef * weights

            weights = weights - self.get_learning_rate(i + 1) * gradient

        self.weights = weights
        self.mse = ((features.dot(weights) - values) ** 2).mean()
        self.rmse = np.sqrt(self.mse)
        self.mae = np.abs(features.dot(weights) - values).mean()
        self.mape = 100 / len_count * np.abs((features.dot(weights) - values) / values).sum()
        self.r2 = 1 - self.mse / np.var(values)

    def get_coef(self):
        return self.weights[1:]

    def get_best_score(self):
        if self.metric == 'mse':
            return self.mse
        elif self.metric == 'mae':
            return self.mae
        elif self.metric == 'rmse':
            return self.rmse
        elif self.metric == 'mape':
            return self.mape
        elif self.metric == 'r2':
            return self.r2
        else:
            return None
    def predict(self, X: pd.DataFrame):
        X.insert(0, '__x0__', 1.0)
        features = X.to_numpy()

        return features.dot(self.weights)


reg = MyLineReg(n_iter=50, learning_rate=0.1, metric='mse')

reg.fit(X, y)
