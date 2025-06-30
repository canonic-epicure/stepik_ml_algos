import random
import pandas as pd
import numpy as np
import copy

from sklearn.metrics import roc_auc_score

from sklearn.datasets import make_regression, make_classification
# X, y = make_regression(n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42)
# X = pd.DataFrame(X)
# y = pd.Series(y)
# X.columns = [f'col_{col}' for col in X.columns]

X, y = make_classification(n_samples=10, n_features=5, n_informative=2, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]


# from sklearn.datasets import load_diabetes
# data = load_diabetes(as_frame=True)
# X, y = data['data'], data['target']

# from temp import X, y, predictX
# X=pd.DataFrame(X[1:], columns=X[0])
# y=pd.Series(y)
# predictX=pd.DataFrame(predictX[1:], columns=predictX[0])

# X, y = make_regression(n_samples=150, n_features=14, n_informative=10, noise=15, random_state=42)
# X = pd.DataFrame(X).round(2)
# y = pd.Series(y)
# X.columns = [f'col_{col}' for col in X.columns]
# test = X.sample(20, random_state=42)

# region helpers
def get_metric_roc_auc(pred, targ):
    pairs = np.stack(
        (
            np.round(pred, 10),
            targ
        ),
        axis=1
    )
    sorted_indicies = np.argsort(pairs[:, 0], axis=0, kind='mergesort')
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
            roc_auc_sum += count_neg * (count_positive_above + count_pos / 2)

        count_positive_above += count_pos

        i = j

    return roc_auc_sum / (count_positive_above * (len - count_positive_above))

def get_entropy(targets):
    if targets.size == 0:
        return 0

    zeros = np.where(targets == 0, 1, 0).sum()

    ones = targets.size - zeros

    prob_zero = zeros / targets.size
    prob_ones = ones / targets.size

    return -((0 if prob_zero == 0 else prob_zero * np.log2(prob_zero)) + (0 if prob_ones == 0 else prob_ones * np.log2(prob_ones)))

def get_gini(targets):
    if targets.size == 0:
        return 1

    zeros = np.where(targets == 0, 1, 0).sum()

    ones = targets.size - zeros

    prob_zero = zeros / targets.size
    prob_ones = ones / targets.size

    return 1 - prob_zero ** 2 - prob_ones **2

def get_i(targets, criterion):
    return get_gini(targets) if criterion == 'gini' else get_entropy(targets) if criterion == 'entropy' else get_mse(targets) if criterion == 'mse' else (_ for _ in ()).throw(ValueError("Unknown criterion"))

def get_info_gain(features, targets, split):
    initial_entropy = get_entropy(targets)

    sorted_idx = np.argsort(features)
    sorted = features[ sorted_idx ]

    idx = np.searchsorted(sorted, split, side='right')

    left_idx = sorted_idx[:idx]
    left_entropy = get_entropy(targets[ left_idx ])

    right_idx = sorted_idx[idx:]
    right_entropy = get_entropy(targets[ right_idx])

    return initial_entropy - (left_entropy * left_idx.size / targets.size + right_entropy * right_idx.size / targets.size)

def get_gini_gain(features, targets, split):
    initial_gini = get_gini(targets)

    sorted_idx = np.argsort(features)
    sorted = features[ sorted_idx ]

    idx = np.searchsorted(sorted, split, side='right')

    left_idx = sorted_idx[:idx]
    left_gini = get_gini(targets[ left_idx ])

    right_idx = sorted_idx[idx:]
    right_gini = get_gini(targets[ right_idx])

    return initial_gini - (left_gini * left_idx.size / targets.size + right_gini * right_idx.size / targets.size)

def get_mse(targets):
    if targets.size == 0:
        return 0

    return np.var(targets)

def get_mse_gain(features, targets, split):
    initial_mse = get_mse(targets)

    sorted_idx = np.argsort(features)
    sorted = features[ sorted_idx ]

    idx = np.searchsorted(sorted, split, side='right')

    left_idx = sorted_idx[:idx]
    left_mse = get_mse(targets[ left_idx ])

    right_idx = sorted_idx[idx:]
    right_mse = get_mse(targets[ right_idx])

    return initial_mse - (left_mse * left_idx.size / targets.size + right_mse * right_idx.size / targets.size)

def get_best_split(X: pd.DataFrame, y, splits, criterion):
    targets = y.to_numpy()

    col_name = None
    split_value = 0
    ig = 0

    for col in X.columns:
        features = X[ col ].to_numpy()

        if splits[ col ].size > 0:
            delimiters = splits[ col ]
        else:
            unique = np.unique(features)

            delimiters = (unique[0:-1] + unique[1:]) / 2

            if delimiters.size == 0:
                delimiters = unique

        gains = np.array(
            list(map(
                lambda delimeter:
                    get_info_gain(features, targets, delimeter) if criterion == 'entropy'
                    else get_gini_gain(features, targets, delimeter) if criterion == 'gini'
                    else get_mse_gain(features, targets, delimeter) if criterion == 'mse'
                    else (_ for _ in ()).throw(ValueError("Unknown criterion")),
                delimiters
            ))
        )

        gains_unique, gains_unique_idx = np.unique(gains, return_index=True)

        delim_sorted_idx = np.argsort(gains_unique, kind='mergesort')
        max_gain = gains_unique[ delim_sorted_idx[ -1 ] ]

        if (col_name == None or ig < max_gain):
            col_name = col
            split_value = delimiters[ gains_unique_idx[ delim_sorted_idx[ -1 ] ] ]
            ig = max_gain

    return col_name, split_value, ig


def for_each_leaf(node, func, level=0):
    if (node[ 'type' ] == 'leaf'):
        func(node, level)
    else:
        for_each_leaf(node[ 'left' ], func, level + 1)
        for_each_leaf(node[ 'right' ], func, level + 1)


def for_each_node(node, func, level=0, side=None):
    if (node[ 'type' ] == 'leaf'):
        if func(node, level, side) == False:
            return
    else:
        func(node, level, side)
        for_each_node(node[ 'left' ], func, level + 1, 'left')
        for_each_node(node[ 'right' ], func, level + 1, 'right')

# endregion

#region Linear regression
class MyLineReg:
    def __init__(self, n_iter=100, learning_rate=0.1, metric=None, reg=None, l1_coef=0.0, l2_coef=0.0, sgd_sample=None, random_state = 42):
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
        features = np.insert(X.to_numpy(), 0, 1, 1)

        return features.dot(self.weights)
#endregion

#region Tree regression
class MyTreeReg:
    def __init__(self, max_depth=5, min_samples_split=2, max_leafs=20, bins=None, criterion='mse', total_n=0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.leafs_cnt = 0
        self.root = None
        self.bins = bins
        self.splits = None
        self.criterion = criterion
        self.fi = {}
        self.total_n = total_n

    def __str__(self):
        return f'MyTreeReg class: max_depth={ self.max_depth }, min_samples_split={ self.min_samples_split }, max_leafs={ self.max_leafs }'

    def build_tree(self, X, y, leaf_count, depth):
        count = X.to_numpy().shape[-2]

        targets_sum = y.to_numpy().sum()

        if (
            depth > 0 and (
                count == 1 #or targets_sum == 0 or targets_sum == count
                or self.leafs_cnt + 1 >= self.max_leafs or depth >= self.max_depth or count < self.min_samples_split
            )
        ):
            self.leafs_cnt += 1

            return { 'type' : 'leaf', 'predict' : y.to_numpy().mean() }
        else:
            col_name, split_value, ig = get_best_split(X, y, self.splits, self.criterion)

            features = X[ col_name ].to_numpy()
            targets = y.to_numpy()

            sorted_idx = np.argsort(features, kind='mergesort')
            sorted = features[ sorted_idx ]

            idx = np.searchsorted(sorted, split_value, side='right')

            if idx == 0 or idx == features.shape[-1]:
                self.leafs_cnt += 1

                return { 'type' : 'leaf', 'predict' : y.to_numpy().mean() }

            lefts_idx = sorted_idx[ :idx ]
            rights_idx = sorted_idx[ idx: ]

            # reserve one node for the right branch
            self.leafs_cnt += 1

            left = self.build_tree(X.iloc[ lefts_idx ], y.iloc[ lefts_idx ], leaf_count + 2, depth + 1)

            # release the reserve
            self.leafs_cnt -= 1

            return {
                'type' : 'node',
                'col_name' : col_name,
                'split_value' : split_value,
                'left' : left,
                'right' : self.build_tree(X.iloc[ rights_idx ], y.iloc[ rights_idx ], leaf_count + 2, depth + 1),
                'importance' : features.size / self.total_n * (
                    get_i(targets, self.criterion)
                    - lefts_idx.size / features.size * get_i(y.iloc[ lefts_idx ], self.criterion)
                    - rights_idx.size / features.size * get_i(y.iloc[ rights_idx ], self.criterion)
                )
            }

    def fit(self, X, y):
        def get_unique_splits(X):
            unique = np.unique(X.to_numpy())

            return (unique[0:-1] + unique[1:]) / 2

        def get_histogram_splits(X):
            histogram_counts, histogram_boundaries = np.histogram(X.to_numpy(), bins=self.bins)

            return histogram_boundaries[ 1:-1 ]

        def get_splits(X):
            unique_splits = get_unique_splits(X)

            if self.bins == None or unique_splits.size <= self.bins - 1:
                return np.array([])
            else:
                return get_histogram_splits(X)

        for col in X.columns:
            self.fi[ col ] = 0

        self.splits = X.apply(get_splits, axis=0)

        self.leafs_cnt = 0
        self.root = self.build_tree(X, y, 0, 0)

        def count(node, _, __):
            if node[ 'type' ] == 'node':
                self.fi[ node[ 'col_name'] ] += node[ 'importance' ]

        for_each_node(self.root, count)

    def print_tree(self):
        def printer(node, level, side):
            indent_str = ' ' * (level * 4)

            if (node[ 'type' ] == 'node'):
                print(f"{ indent_str }{ node[ 'col_name' ] } > { node[ 'split_value' ] }")
            else:
                print(f"{ indent_str }leaf_{ side } = { node[ 'predict' ] }")

        print(self)
        for_each_node(self.root, printer)

    def predict(self, X):
        def predict_single(x):
            prob = 0

            def predict(node):
                nonlocal prob

                if node[ 'type' ] == 'leaf':
                    prob = node[ 'predict' ]
                else:
                    value = x[ node[ 'col_name' ] ]

                    if value > node[ 'split_value' ]:
                        predict(node[ 'right' ])
                    else:
                        predict(node[ 'left' ])

            predict(self.root)

            return prob

        return X.apply(predict_single, axis=1).to_numpy()
#endregion

#region kNN regression
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
#endregion

class MyBaggingReg:
    def __init__(self, estimator=None, n_estimators=10, max_samples=0.5, random_state=42, oob_score=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state

        self.estimators = []
        self.oob_score = oob_score
        self.oob_score_ = 0

    def __str__(self):
        return f'MyBaggingReg class: estimator={ self.estimator }, n_estimators={ self.n_estimators }, max_samples={ self.max_samples }, random_state={ self.random_state }'

    def fit(self, X, y):
        random.seed(self.random_state)

        rows = range(X.shape[-2])
        rows_numpy = np.array(rows)

        oob_predictions = list(map(lambda _: [], rows))

        rows_indicies = list(
            map(
                lambda i: random.choices(rows, k=round(X.shape[-2] * self.max_samples)),
                range(self.n_estimators)
            )
        )

        self.estimator.total_n = X.shape[ -2 ]

        for i in range(self.n_estimators):
            rows_idx = rows_indicies[ i ]

            model = copy.copy(self.estimator)

            model.fit(X.loc[ rows_idx, : ], y.loc[ rows_idx ])

            rows_mask = np.ones(len(rows), dtype=bool)
            rows_mask[ rows_idx ] = False

            predictions = model.predict(X.loc[ rows_mask, : ])

            for idx, value in enumerate(rows_numpy[ rows_mask ]):
                oob_predictions[ value ].append(predictions[ idx ])

            self.estimators.append(model)

        oob_predictions = np.array(list(map(lambda pred: np.nan if len(pred) == 0 else np.array(pred).mean(), oob_predictions)))

        oob_not_nan_idx = ~np.isnan(oob_predictions)

        targets = y.to_numpy()

        pred = oob_predictions[ oob_not_nan_idx ]
        targ = targets[ oob_not_nan_idx ]

        if self.oob_score == 'mse':
            self.oob_score_ = ((pred - targ) ** 2).mean()
        elif self.oob_score == 'rmse':
            self.oob_score_ = np.sqrt(((pred - targ) ** 2).mean())
        elif self.oob_score == 'mae':
            self.oob_score_ = np.abs(pred - targ).mean()
        elif self.oob_score == 'mape':
            self.oob_score_ = 100 / pred.shape[-1] * np.abs((targ - pred) / targ).sum()
        elif self.oob_score == 'r2':
            self.oob_score_ = 1 - ((pred - targ) ** 2).mean() / targ.var()
        elif self.oob_score != None:
            raise f"Unknown metric: { self.oob_score }"

    def predict(self, X):
        res = np.array(list(map(
            lambda estimator: estimator.predict(X),
            self.estimators
        )))

        return res.mean(axis=0)

clf = MyBaggingReg(estimator=MyLineReg())

clf.fit(X, y)

# # list(map(lambda tree: tree.print_tree(), clf.trees))
#
# # print(clf.predict(test).sum())
#
# print(clf.oob_score_)
#
