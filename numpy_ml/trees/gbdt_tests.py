#

#################################################
#### sklearn
import numpy as np
from sklearn.datasets import load_breast_cancer  # 经典二分类
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
X, y = load_breast_cancer(return_X_y=True)
from sklearn.metrics import *
np.random.seed(0)

######################
random_order = list(range(len(X)))
np.random.shuffle(random_order)
train_samples = int(len(random_order)*0.8,)
X_train = X[random_order[:train_samples]]
X_test =  X[random_order[train_samples:]]

y_train = y[random_order[:train_samples]]
y_test =  y[random_order[train_samples:]]

#################################################
# run

from importlib import reload
from numpy_ml.trees import gbdt
reload(gbdt)
from numpy_ml.trees.gbdt import *


gt = GradientBoostedDecisionTree(n_iter=10,
                                 max_depth=3,
                                 classifier=True,
                                 learning_rate=1,
                                 loss="crossentropy")
gt.fit(X_train,y_train)

gt.predict(X_test)

accuracy_score(y_test,gt.predict(X_test))
# accuracy_score(y_test,gt.predict(X_test))
# Out[6]: 0.9473684210526315






