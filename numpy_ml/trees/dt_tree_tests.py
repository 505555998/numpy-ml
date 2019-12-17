# 下午测试：
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

from importlib import reload
from numpy_ml.trees import dt
reload(dt)
from numpy_ml.trees.dt import *

mine = DecisionTree(
    classifier=True, max_depth=3, criterion="entropy",seed=0
)


mine.fit(X_train,y_train)


# ****************************************************************************************************
# depth: 0 no_split_data: (455, 30)
# depth 1 split by  (20, 16.795) left train (295, 30) right train (160, 30)
# ****************************************************************************************************
# depth: 1 no_split_data: (295, 30)
# depth 2 split by  (27, 0.13579999999999998) left train (259, 30) right train (36, 30)
# ****************************************************************************************************
# depth: 2 no_split_data: (259, 30)
# depth 3 split by  (1, 21.435000000000002) left train (212, 30) right train (47, 30)

# depth 3 left <numpy_ml.trees.dt.Leaf object at 0x1a3d90c080>
# depth 3 right <numpy_ml.trees.dt.Leaf object at 0x10eb02198>
# depth 3 left <numpy_ml.trees.dt.Node object at 0x1a3d97fe80>
# depth 3 right <numpy_ml.trees.dt.Leaf object at 0x1a3d97fac8>
# depth 3 left <numpy_ml.trees.dt.Node object at 0x1a3d97fa90>
# depth 3 right <numpy_ml.trees.dt.Leaf object at 0x1a3d915eb8>




#### 原始数据，探索
# array 多个条件
# mask = (X_train[:,20]<=16.795) & (X_train[:,27]<=0.13579) & (X_train[:,1]<=21.435)
# len(mask) # 455
# sum(mask) # 212
# # 位置
# np.argwhere(mask)
#
# # 直接拿数据
# ##### 最后一层：
# # left
# y_train[mask]
# # right
# a1 = y_train[(X_train[:,20]<=16.795) & (X_train[:,27]<=0.13579) & (X_train[:,1]>21.435)]
# np.bincount(a1)/len(a1)  #array([0.08510638, 0.91489362])



###############
# 评估
roc_auc_score(y_test,mine.predict_class_probs(X_test)[:,1],)
# 0.9865259740259741

###sklearn

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

dt = DecisionTreeClassifier(
                criterion="entropy",max_depth=3,
                splitter="best",
                random_state=0,)

dt.fit(X_train,y_train)
roc_auc_score(y_test,dt.predict_proba(X_test)[:,1],)
# 0.9865259740259741



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


