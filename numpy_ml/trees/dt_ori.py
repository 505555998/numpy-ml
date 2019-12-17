import numpy as np


class Node:
    # 分割，内部节点
    def __init__(self, left, right, rule):
        self.left = left
        self.right = right
        self.feature = rule[0]
        self.threshold = rule[1]


class Leaf:
    # 分割后的叶节点
    def __init__(self, value):
        """
        `value` is an array of class probabilities if classifier is True, else
        the mean of the region
        """
        self.value = value


class DecisionTree:
    def __init__(
        self,
        classifier=True,
        max_depth=None,
        n_feats=None,
        criterion="entropy",
        seed=None,
    ):
        """
        A decision tree model for regression and classification problems.

        Parameters
        ----------
        classifier : bool
            Whether to treat target values as categorical (classifier =
            True) or continuous (classifier = False). Default is True.
        max_depth: int or None
            The depth at which to stop growing the tree. If None, grow the tree
            until all leaves are pure. Default is None.
        n_feats : int
            Specifies the number of features to sample on each split. If None,
            use all features on each split. Default is None.
        criterion : {'mse', 'entropy', 'gini'}
            The error criterion to use when calculating splits. When
            `classifier` is False, valid entries are {'mse'}. When `classifier`
            is True, valid entries are {'entropy', 'gini'}. Default is
            'entropy'.
        seed : int or None
            Seed for the random number generator. Default is None.
        """

        # 初始值：
        self.depth = 0
        self.root = None
        self.seed = seed
        self.n_feats = n_feats
        self.criterion = criterion
        self.classifier = classifier
        self.max_depth = max_depth if max_depth else np.inf

        if not classifier and criterion in ["gini", "entropy"]:
            raise ValueError(
                "{} is a valid criterion only when classifier = True.".format(criterion)
            )
        if classifier and criterion == "mse":
            raise ValueError("`mse` is a valid criterion only when classifier = False.")

    def fit(self, X, Y):
        """
        Fit a binary decision tree to a dataset.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            The training data of `N` examples, each with `M` features
        Y : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
            An array of integer class labels for each example in `X` if
            self.classifier = True, otherwise the set of target values for
            each example in `X`.

        `N` examples, each with `M` features

        """
        self.n_classes = max(Y) + 1 if self.classifier else None
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow(X, Y)

    def predict(self, X):
        """
        Use the trained decision tree to classify or predict the examples in `X`.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            The training data of `N` examples, each with `M` features

        Returns
        -------
        preds : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
            The integer class labels predicted for each example in `X` if
            self.classifier = True, otherwise the predicted target values.
        """
        return np.array([self._traverse(x, self.root) for x in X])

    def predict_class_probs(self, X):
        """
        Use the trained decision tree to return the class probabilities for the
        examples in `X`.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            The training data of `N` examples, each with `M` features

        Returns
        -------
        preds : :py:class:`ndarray <numpy.ndarray>` of shape `(N, n_classes)`
            The class probabilities predicted for each example in `X`.
        """
        assert self.classifier, "`predict_class_probs` undefined for classifier = False"
        return np.array([self._traverse(x, self.root, prob=True) for x in X])

    def _grow(self, X, Y):
        if self.seed:
            np.random.seed(self.seed)

        # if all labels are the same, return a leaf
        # 递归生长

        # 停止条件：
        if len(set(Y)) == 1: # 完全是0 或者是1
            if self.classifier:
                prob = np.zeros(self.n_classes)
                # prob[Y[0]] = 1.0
                # 其实Y[0]==1 或者 0
                prob[Y[0]] = 1.0
            return Leaf(prob) if self.classifier else Leaf(Y[0])  # Leaf(Y[0]) 是回归
        # if we have reached max_depth, return a leaf
        if self.depth >= self.max_depth:
            # 如果是回归：
            v = np.mean(Y, axis=0)

            # 如果是分类：
            # 如果是2分类的话，其实也是均值，
            # 0：10，1：20 。mean=2/3

            # 但是这块要表述成：0-1-2-3-4... 的对应概率
            if self.classifier:
                v = np.bincount(Y, minlength=self.n_classes) / len(Y)
            return Leaf(v)


        N, M = X.shape
        print("*"*100)
        print("before split .depth:",self.depth,"no_split_data:",X.shape)

        self.depth += 1

        feat_idxs = np.random.choice(M, self.n_feats, replace=False)

        # greedily select the best split according to `criterion`
        feat, thresh = self._segment(X, Y, feat_idxs)

        l = np.argwhere(X[:, feat] <= thresh).flatten()
        r = np.argwhere(X[:, feat] > thresh).flatten()
        print("depth", self.depth, "split by ",(feat,thresh),"left train",X[l, :].shape,"right train",X[r, :].shape)
        # depth 1 311
        # depth 2 273
        # depth 3 246

        # grow the children that result from the split
        left = self._grow(X[l, :], Y[l])
        print("depth",self.depth,"left",left,) #left.value,len(left.value))

        # 这块的self.depth ==max_depth，因为左树贪婪生长满足了depth
        right = self._grow(X[r, :], Y[r])
        print("depth", self.depth, "right", right,right.value,len(right.value)) #
        return Node(left, right, (feat, thresh))

    def _segment(self, X, Y, feat_idxs):
        """
        Find the optimal split rule (feature index and split threshold) for the
        data according to `self.criterion`.
        # 所有特征，所有阈值（这里用的错位均值）
        """
        best_gain = -np.inf
        split_idx, split_thresh = None, None
        for i in feat_idxs:
            vals = X[:, i]
            levels = np.unique(vals)
            # thresholds 是错位均值
            thresholds = (levels[:-1] + levels[1:]) / 2 if len(levels) > 1 else levels
            gains = np.array([self._impurity_gain(Y, t, vals) for t in thresholds])

            if gains.max() > best_gain:
                split_idx = i
                best_gain = gains.max()
                split_thresh = thresholds[gains.argmax()]

        return split_idx, split_thresh

    def _impurity_gain(self, Y, split_thresh, feat_values):
        """
        Compute the impurity gain associated with a given split.

        IG(split) = loss(parent) - weighted_avg[loss(left_child), loss(right_child)]
        """
        if self.criterion == "entropy":
            loss = entropy
        elif self.criterion == "gini":
            loss = gini
        elif self.criterion == "mse":
            loss = mse

        parent_loss = loss(Y)

        # generate split
        # index: 返回的是index 可计算长度  也可以拿出来数据
        left = np.argwhere(feat_values <= split_thresh).flatten()
        right = np.argwhere(feat_values > split_thresh).flatten()

        if len(left) == 0 or len(right) == 0:
            return 0

        # compute the weighted avg. of the loss for the children
        n = len(Y)
        n_l, n_r = len(left), len(right)
        e_l, e_r = loss(Y[left]), loss(Y[right])
        child_loss = (n_l / n) * e_l + (n_r / n) * e_r

        # impurity gain is difference in loss before vs. after split
        ig = parent_loss - child_loss
        return ig

    def _traverse(self, X, node, prob=False):
        if isinstance(node, Leaf):
            if self.classifier:
                return node.value if prob else node.value.argmax()
            return node.value
        if X[node.feature] <= node.threshold:
            return self._traverse(X, node.left, prob)
        return self._traverse(X, node.right, prob)


def mse(y):
    """
    Mean squared error for decision tree (ie., mean) predictions
    """
    return np.mean((y - np.mean(y)) ** 2)


def entropy(y):
    """
    Entropy of a label sequence
    也就是交叉熵损失：
    """
    hist = np.bincount(y)
    ps = hist / np.sum(hist)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])


def gini(y):
    """
    Gini impurity (local entropy) of a label sequence
    """
    hist = np.bincount(y)
    N = np.sum(hist)
    return 1 - sum([(i / N) ** 2 for i in hist])
