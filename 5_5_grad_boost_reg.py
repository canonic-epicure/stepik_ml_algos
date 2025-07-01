import random
import pandas as pd
import numpy as np
import copy

from sklearn.metrics import roc_auc_score

from sklearn.datasets import make_regression, make_classification
X, y = make_regression(n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]

# X, y = make_classification(n_samples=10, n_features=5, n_informative=2, random_state=42)
# X = pd.DataFrame(X)
# y = pd.Series(y)
# X.columns = [f'col_{col}' for col in X.columns]


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
    return get_gini(targets) if criterion == 'gini' \
        else get_entropy(targets) if criterion == 'entropy' \
        else get_mse(targets) if criterion == 'mse' \
        else (_ for _ in ()).throw(ValueError("Unknown criterion"))

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

#region Tree regression
class MyTreeReg:
    def __init__(self, max_depth=5, min_samples_split=2, max_leafs=20, bins=16, criterion='mse', total_n=0):
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

            return { 'type' : 'leaf', 'index' : X.index, 'predict' : y.to_numpy().mean() }
        else:
            col_name, split_value, ig = get_best_split(X, y, self.splits, self.criterion)

            features = X[ col_name ].to_numpy()
            targets = y.to_numpy()

            sorted_idx = np.argsort(features, kind='mergesort')
            sorted = features[ sorted_idx ]

            idx = np.searchsorted(sorted, split_value, side='right')

            if idx == 0 or idx == features.shape[-1]:
                self.leafs_cnt += 1

                return { 'type' : 'leaf', 'index' : X.index, 'predict' : y.to_numpy().mean() }

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

class MyBoostReg:
    def __init__(
        self, n_estimators=10, learning_rate=0.1, max_depth=5, min_samples_split=2, max_leafs=20, bins=16, loss='MSE', metric=None,
        max_features=0.5, max_samples=0.5, random_state=42,
        reg=0.1
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.best_score = 0
        self.metric = metric
        self.reg = reg

        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins

        self.pred_0 = 0
        self.trees = []
        self.fi = {}
        self.leafs_cnt = 0

    def __str__(self):
        return f'MyBoostReg class: n_estimators={ self. n_estimators }, learning_rate={ self. learning_rate }, max_depth={ self. max_depth }, min_samples_split={ self. min_samples_split }, max_leafs={ self. max_leafs }, bins={ self. bins }'

    def get_pred_0(self, targets: np.array):
        if self.loss == 'MSE':
            return targets.mean()
        elif self.loss == 'MAE':
            return np.median(targets)

    def get_grad(self, targets, predictions):
        if self.loss == 'MSE':
            return 2 * (predictions - targets)
        elif self.loss == 'MAE':
            return np.sign(predictions - targets)

    def get_loss(self, targets, predictions):
        if self.loss == 'MSE':
            return ((predictions - targets) ** 2).mean()
        elif self.loss == 'MAE':
            return np.abs(predictions - targets).mean()

    def get_learning_rate(self, iteration: int) -> float:
        if callable(self.learning_rate):
            return self.learning_rate(iteration)
        else:
            return self.learning_rate

    def fit(self, X, y, X_eval=None, y_eval=None, early_stopping=None, verbose=None):
        random.seed(self.random_state)

        targets = y.to_numpy()

        self.pred_0 = self.get_pred_0(targets)

        self.total_n = X.shape[ -2 ]
        for col in X.columns:
            self.fi[ col ] = 0

        columns = list(X.columns)
        rows = range(X.shape[-2])

        pred = np.array([ self.pred_0 ] * X.shape[ -2 ])

        validation_score_tracking = []
        prev_validation_score = None

        for i in range(self.n_estimators):
            cols_idx = random.sample(columns, round(X.shape[-1] * self.max_features))
            rows_idx = random.sample(rows, round(X.shape[-2] * self.max_samples))

            model = MyTreeReg(max_depth=self.max_depth, min_samples_split=self.min_samples_split, max_leafs=self.max_leafs, bins=self.bins, total_n=self.total_n)

            self.trees.append(model)

            model.fit(X.loc[ rows_idx, cols_idx ], pd.Series(-self.get_grad(targets, pred)[ rows_idx ]))

            def process(node, _):
                diff = y.loc[ node[ 'index' ] ] - pred[ node[ 'index' ] ]
                node[ 'predict' ] = self.get_pred_0(diff) + self.reg * self.leafs_cnt

            for_each_leaf(model.root, process)

            self.leafs_cnt += model.leafs_cnt

            pred = pred + self.get_learning_rate(i + 1) * model.predict(X)

            if verbose and (i % verbose) == 0:
                print(f'{ i } loss:', self.get_loss(self.predict(X), targets))

            if early_stopping is not None and X_eval is not None and y_eval is not None:
                validation_score = self.get_score(self.predict(X_eval), y_eval.to_numpy())

                if prev_validation_score is not None:
                    # metric has not improved as binary value
                    validation_score_tracking.append(validation_score >= prev_validation_score)
                    if len(validation_score_tracking) > early_stopping:
                        validation_score_tracking.pop()

                    if np.array(validation_score_tracking).sum() == early_stopping:
                        self.trees = self.trees[:-early_stopping]
                        break

                prev_validation_score = validation_score

        for tree in self.trees:
            for col in X.columns:
                self.fi[ col ] += tree.fi[ col ] if col in tree.fi else 0

                self.fi[ col ] *= 1.0

        if early_stopping is not None and X_eval is not None and y_eval is not None:
            self.best_score = self.get_score(self.predict(X_eval), y_eval.to_numpy())
        else:
            self.best_score = self.get_score(self.predict(X), targets)

    def get_score(self, pred, targets):
        if self.metric == 'MSE':
            return ((pred - targets) ** 2).mean()
        elif self.metric == 'RMSE':
            return np.sqrt(((pred - targets) ** 2).mean())
        elif self.metric == 'MAE':
            return np.abs(pred - targets).mean()
        elif self.metric == 'MAPE':
            return 100 / pred.shape[-1] * np.abs((targets - pred) / targets).sum()
        elif self.metric == 'R2':
            return 1 - ((pred - targets) ** 2).mean() / targets.var()
        elif self.metric == None:
            return self.get_loss(pred, targets)
        else:
            raise f"Unknown metric: { self.metric }"

    def predict(self, X):
        res = np.array([ self.get_learning_rate(idx + 1) * tree.predict(X) for idx, tree in enumerate(self.trees) ])

        return res.sum(axis=0) + self.pred_0

reg = MyBoostReg()

reg.fit(X, y)

# # list(map(lambda tree: tree.print_tree(), reg.trees))
#
# # print(reg.predict(test).sum())
#
# print(reg.oob_score_)
#
