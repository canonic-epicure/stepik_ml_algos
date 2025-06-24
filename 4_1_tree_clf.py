import pandas as pd
import numpy as np

df = pd.read_csv('data/data_banknote_authentication.txt', header=None)
df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'target']
X, y = df.iloc[:,:4], df['target']

#-------------------------------------------
def get_best_split(X: pd.DataFrame, y, splits, criterion):

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

        entropy_deltas = np.array(
            list(map(
                lambda delim: get_info_gain(features, targets, delim) if criterion == 'entropy' else get_gini_gain(features, targets, delim),
                delimiters
            ))
        )

        delim_sorted_idx = np.argsort(entropy_deltas)
        max_entropy_delta = entropy_deltas[ delim_sorted_idx[ -1 ] ]

        if (col_name == None or ig < max_entropy_delta):
            col_name = col
            split_value = delimiters[ delim_sorted_idx[ -1 ] ]
            ig = max_entropy_delta

    return col_name, split_value, ig


#-------------------------------------------
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

#-------------------------------------------
class MyTreeClf:
    def __init__(self, max_depth=5, min_samples_split=2, max_leafs=20, bins=None, criterion='entropy'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.leafs_cnt = 0
        self.root = None
        self.bins = bins
        self.splits = None
        self.criterion = criterion

    def __str__(self):
        return f'MyTreeClf class: max_depth={ self.max_depth }, min_samples_split={ self.min_samples_split }, max_leafs={ self.max_leafs }'

    def build_tree(self, X, y, leaf_count, depth):
        count = X.to_numpy().shape[-2]

        targets_sum = y.to_numpy().sum()


        if (
            depth > 0 and (
                count == 1 or targets_sum == 0 or targets_sum == count
                or self.leafs_cnt + 1 >= self.max_leafs or depth >= self.max_depth or count < self.min_samples_split
            )
        ):
            self.leafs_cnt += 1

            return { 'type' : 'leaf', 'one_prob' : y.to_numpy().mean() }
        else:
            col_name, split_value, ig = get_best_split(X, y, self.splits, self.criterion)

            features = X[ col_name ].to_numpy()

            sorted_idx = np.argsort(features)
            sorted = features[ sorted_idx ]

            idx = np.searchsorted(sorted, split_value, side='right')

            if idx == 0 or idx == features.shape[-1]:
                self.leafs_cnt += 1

                return { 'type' : 'leaf', 'one_prob' : y.to_numpy().mean() }

            lefts_idx = sorted_idx[ :idx ]
            rights_idx = sorted_idx[ idx: ]

            return {
                'type' : 'node',
                'col_name' : col_name,
                'split_value' : split_value,
                'left' : self.build_tree(X.iloc[ lefts_idx ], y.iloc[ lefts_idx ], leaf_count + 2, depth + 1),
                'right' : self.build_tree(X.iloc[ rights_idx ], y.iloc[ rights_idx ], leaf_count + 2, depth + 1)
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

        self.splits = X.apply(get_splits, axis=0)

        self.leafs_cnt = 0
        self.root = self.build_tree(X, y, 0, 0)

    def print_tree(self):
        def printer(node, level, side):
            indent_str = ' ' * (level * 4)

            if (node[ 'type' ] == 'node'):
                print(f'{ indent_str }{ node[ 'col_name' ] } > { node[ 'split_value' ] }')
            else:
                print(f'{ indent_str }leaf_{ side } = { node[ 'one_prob' ] }')

        for_each_node(self.root, printer)

    def predict(self, X):
        return (self.predict_proba(X) > 0.5) * 1

    def predict_proba(self, X):
        def predict_single(x):
            prob = 0

            def predict(node):
                nonlocal prob

                if node[ 'type' ] == 'leaf':
                    prob = node[ 'one_prob' ]
                else:
                    value = x[ node[ 'col_name' ] ]

                    if value > node[ 'split_value' ]:
                        predict(node[ 'right' ])
                    else:
                        predict(node[ 'left' ])

            predict(self.root)

            return prob

        return X.apply(predict_single, axis=1)

clf = MyTreeClf(max_depth=5, min_samples_split=200, max_leafs=10, bins=4)

clf.fit(X, y)

count=0
sum=0

def cc(node):
    global count, sum
    count = count + 1
    sum += node[ 'one_prob' ]

for_each_leaf(clf.root, lambda node, __: cc(node) )

print(count, sum)

clf.print_tree()

clf.predict_proba(X)