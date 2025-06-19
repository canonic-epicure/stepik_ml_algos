import \
    random

import \
    numpy as np
import \
    pandas as pd
from sklearn.datasets import \
    make_regression

X, y = make_regression(n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]

class MyLogReg:
    # `metric` is one of [ `accuracy`, `precision`, `recall`, `f1`, `roc_auc` ],
    # `reg` is one of [ `l1`, `l2`, `elasticnet` ],
    def __init__(self, n_iter, learning_rate, metric = None, reg=None, l1_coef=0.0, l2_coef=0.0, sgd_sample=None, random_state = 42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.metric = metric
        self.values = None
        self.features = None
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def get_learning_selection_idx(self, features):
        if self.sgd_sample == None:
            return None
        else:
            count = self.sgd_sample if isinstance(self.sgd_sample, int) else round(self.sgd_sample * features.shape[0])
            return random.sample(range(features.shape[0]), count)

    def __str__(self):
        return f'MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'

    def get_learning_rate(self, iteration):
        if callable(self.learning_rate):
            return self.learning_rate(iteration)
        else:
            return self.learning_rate

    def get_metric_accuracy(self, features, values):
        prediction = 1 / (1 + np.e ** (-features.dot(self.weights)))

        return ((values - (prediction >= 0.5) * 1) == 0).mean()

    def get_metric_precision(self, features, values):
        prediction = 1 / (1 + np.e ** (-features.dot(self.weights)))

        prediction_cls = (prediction >= 0.5) * 1

        tp = np.logical_and(prediction_cls == 1, values == 1).sum()
        fp = np.logical_and(prediction_cls == 1, values == 0).sum()

        return tp / (tp + fp)

    def get_metric_recall(self, features, values):
        prediction = 1 / (1 + np.e ** (-features.dot(self.weights)))

        prediction_cls = (prediction >= 0.5) * 1

        tp = np.logical_and(prediction_cls == 1, values == 1).sum()
        fn = np.logical_and(prediction_cls == 0, values == 1).sum()

        return tp / (tp + fn)

    def get_metric_f1(self, features, values):
        precision = self.get_metric_precision(features, values)
        recall = self.get_metric_recall(features, values)

        return 2 * precision * recall / (precision + recall)

    def get_metric_roc_auc(self, features, values):
        pairs = np.stack(
            (
                np.round(1 / (1 + np.e ** (-features.dot(self.weights))), 10),
                values
            ),
            axis=1
        )
        sorted_indicies = np.argsort(pairs[:, 0], axis=0)
        pairs = pairs[sorted_indicies][::-1]

        len = pairs.shape[-2]

        count_positive_above = 0
        roc_auc_sum = 0

        i = 0

        while i < len:
            count_pos = 0
            count_neg = 0

            j = i

            while j == i or j < len and pairs[j, 0] == pairs[j - 1, 0]:
                if (pairs[j, 1] == 1):
                    count_pos += 1
                else:
                    count_neg += 1

                j += 1

            if (count_neg > 0):
                roc_auc_sum += count_neg * count_positive_above + count_pos / 2

            count_positive_above += count_pos

            i = j

        return roc_auc_sum / (count_positive_above * (len - count_positive_above))

    def get_metric(self, features, values):
        if self.metric == 'accuracy':
            return self.get_metric_accuracy(features, values)
        elif self.metric == 'precision':
            return self.get_metric_precision(features, values)
        elif self.metric == 'recall':
            return self.get_metric_recall(features, values)
        elif self.metric == 'f1':
            return self.get_metric_f1(features, values)
        elif self.metric == 'roc_auc':
            return self.get_metric_roc_auc(features, values)
        else:
            return None

    def get_reg_loss(self):
        if self.reg == 'l1':
            return self.l1_coef * np.abs(self.weights).sum()
        elif self.reg == 'l2':
            return self.l2_coef * (self.weights ** 2).sum()
        elif self.reg == 'elasticnet':
            return self.l1_coef * np.abs(self.weights).sum() + self.l2_coef * (self.weights ** 2).sum()
        else:
            return 0

    def get_reg_grad(self):
        if self.reg == 'l1':
            return self.l1_coef * np.sign(self.weights)
        elif self.reg == 'l2':
            return self.l2_coef * (2 * self.weights)
        elif self.reg == 'elasticnet':
            return self.l1_coef * np.sign(self.weights) + self.l2_coef * (2 * self.weights)
        else:
            return np.zeros(self.weights.shape)

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: bool = False):
        random.seed(self.random_state)
        eps = 1e-15

        features = self.features = np.insert(X.to_numpy(), 0, 1, 1)
        values = self.values = y.to_numpy()

        self.weights = np.array([1.0] * features.shape[-1])

        for i in range(self.n_iter):
            learning_idx = self.get_learning_selection_idx(features)
            learning_features = features if learning_idx == None else features[learning_idx]
            learning_values = values if learning_idx == None else values[learning_idx]

            prediction = 1 / (1 + np.e ** (-learning_features.dot(self.weights)))

            loss_arr = list(map(
                lambda pred, value: value * np.log(pred + eps) + (1 - value) * np.log(1 - pred + eps),
                prediction,
                values
            ))

            loss = -(np.array(loss_arr).mean()) + self.get_reg_loss()

            if verbose and (i % verbose) == 0:
                print(f'{i} | loss={loss}')

            gradient = 1 / learning_features.shape[0] * learning_features.T.dot(prediction - learning_values) + self.get_reg_grad()

            self.weights = self.weights - self.get_learning_rate(i + 1) * gradient

    def get_coef(self):
        return self.weights[1:]

    def predict_proba(self, X: pd.DataFrame):
        features = np.insert(X.to_numpy(), 0, 1, 1)

        return 1 / (1 + np.e ** (-features.dot(self.weights)))

    def predict(self, X: pd.DataFrame):
        return (self.predict_proba(X) >= 0.5) * 1

    def get_best_score(self):
        return self.get_metric(self.features, self.values)

reg = MyLogReg(n_iter=50, learning_rate=0.1, metric='roc_auc')

reg.fit(X, y, 1)

print(reg.get_best_score())