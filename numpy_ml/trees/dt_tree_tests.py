# 下午测试：
#################################################
#### sklearn
import numpy as np
from sklearn.datasets import load_breast_cancer  # 经典二分类
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
X, y = load_breast_cancer(return_X_y=True)
from sklearn.metrics import *


random_order = list(range(len(X)))
np.random.shuffle(random_order)
train_samples = int(len(random_order)*0.8,)
X_train = X[random_order[:train_samples]]
X_test =  X[random_order[train_samples:]]

y_train = y[random_order[:train_samples]]
y_test =  y[random_order[train_samples:]]

#################################################

from importlib import reload
from numpy_ml.trees import dt
reload(dt)
from numpy_ml.trees.dt import *

mine = DecisionTree(
    classifier=True, max_depth=3, criterion="entropy"
)


mine.fit(X_train,y_train)

y_pred = mine.predict(X_test)


mine.predict(X_train)
accuracy_score(y_test,y_pred,)








# #################################################
# # 原始测试：
# # from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# # from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
# import numpy as np
# from sklearn.metrics import accuracy_score, mean_squared_error
# from sklearn.datasets import make_regression
# from sklearn.datasets.samples_generator import make_blobs
# from sklearn.model_selection import train_test_split
# from importlib import reload
# from numpy_ml.trees import dt
# reload(dt)
# from numpy_ml.trees.dt import *
#
#
# np.random.seed(12345)
#
# n_ex = np.random.randint(2, 100)
# n_feats = np.random.randint(2, 100)
# max_depth = np.random.randint(1, 5)
#
# n_classes = 2
# X, Y = make_blobs(n_samples=n_ex, centers=n_classes, n_features=n_feats, random_state=0)
# X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
# mine = DecisionTree(classifier=True, max_depth=max_depth, criterion="entropy")
# # fit 'em
# mine.fit(X, Y)
#
#
# # get preds on training set
# y_pred_mine_test = mine.predict(X_test)
# accuracy_score(Y_test,y_pred_mine_test)


