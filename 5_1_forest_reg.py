import random
import pandas as pd
import numpy as np

from sklearn.datasets import make_regression
# X, y = make_regression(n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42)
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

X, y = make_regression(n_samples=150, n_features=14, n_informative=10, noise=15, random_state=42)
X = pd.DataFrame(X).round(2)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]
test = X.sample(20, random_state=42)

# region tree helpers
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
                lambda delim: get_mse_gain(features, targets, delim),
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

#-------------------------------------------
class MyTreeReg:
    def __init__(self, max_depth=5, min_samples_split=2, max_leafs=20, bins=16, criterion='entropy', total_n=0):
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

        if (
            depth > 0 and (
                count == 1
                or self.leafs_cnt + 1 >= self.max_leafs or depth >= self.max_depth or count < self.min_samples_split
            )
        ):
            self.leafs_cnt += 1

            return { 'type' : 'leaf', 'predict' : y.to_numpy().mean() }
        else:
            col_name, split_value, ig = get_best_split(X, y, self.splits, self.criterion)

            features = X[ col_name ].to_numpy()
            targets = y.to_numpy()

            sorted_idx = np.argsort(features)
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
                    get_mse(targets)
                    - lefts_idx.size / features.size * get_mse(y.iloc[ lefts_idx ])
                    - rights_idx.size / features.size * get_mse(y.iloc[ rights_idx ])
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

class MyForestReg:
    def __init__(self, n_estimators=10, max_features=0.5, max_samples=0.5, random_state=42, max_depth=5, min_samples_split=2, max_leafs=20, bins=16, oob_score=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins

        self.leafs_cnt = 0

        self.trees = []
        self.fi = {}

        self.oob_score = oob_score
        self.oob_score_ = 0

    def __str__(self):
        return f'MyForestReg class: n_estimators={ self.n_estimators }, max_features={ self.max_features }, max_samples={ self.max_samples }, max_depth={ self.max_depth }, min_samples_split={ self.min_samples_split }, max_leafs={ self.max_leafs }, bins={ self.bins }, random_state={ self.random_state }'

    def fit(self, X, y):
        random.seed(self.random_state)

        self.total_n = X.shape[0]
        for col in X.columns:
            self.fi[ col ] = 0

        columns = list(X.columns)
        rows = range(X.shape[-2])
        rows_numpy = np.array(rows)

        oob_predictions = list(map(lambda _: [], rows))

        for i in range(self.n_estimators):
            cols_idx = random.sample(columns, round(X.shape[-1] * self.max_features))
            rows_idx = random.sample(rows, round(X.shape[-2] * self.max_samples))

            tree = MyTreeReg(max_depth=self.max_depth, min_samples_split=self.min_samples_split, max_leafs=self.max_leafs, bins=self.bins, total_n=self.total_n)

            tree.fit(X.loc[ rows_idx, cols_idx ], y.loc[ rows_idx ])

            rows_mask = np.ones(len(rows), dtype=bool)
            rows_mask[ rows_idx ] = False

            predictions = tree.predict(X.loc[ rows_mask, cols_idx ])

            for idx, value in enumerate(rows_numpy[ rows_mask ]):
                oob_predictions[ value ].append(predictions[ idx ])

            self.leafs_cnt += tree.leafs_cnt

            self.trees.append(tree)

        for tree in self.trees:
            for col in X.columns:
                self.fi[ col ] += tree.fi[ col ] if col in tree.fi else 0

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
        res = np.array(
            list(
                map(
                    lambda tree: tree.predict(X),
                    self.trees
                )
            )
        )

        return res.mean(axis=0)

clf = MyForestReg(n_estimators=5, max_features=0.4, max_samples=0.3, max_depth=4, min_samples_split=2, max_leafs=20, bins=16, random_state=42, oob_score='mae')

clf.fit(X, y)

# list(map(lambda tree: tree.print_tree(), clf.trees))

# print(clf.predict(test).sum())

print(clf.oob_score_)

