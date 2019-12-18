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


# shape: (455, 30)
# ****************************************************************************************************
# before split no_split_data: (455, 30)
# depth 1 split by  (22, 105.15) left train 275 264 right train 180 23
# shape: (275, 30)
# ****************************************************************************************************
# before split no_split_data: (275, 30)
# depth 2 split by  (27, 0.13505) left train 259 255 right train 16 9
# shape: (259, 30)
# ****************************************************************************************************
# before split no_split_data: (259, 30)
# depth 3 split by  (13, 48.975) left train 255 253 right train 4 2
# shape: (255, 30)
# depth 3 left <numpy_ml.trees.dt.Leaf object at 0x1a3a9f7b38> left_sample 255 right_sample 4
# shape: (4, 30)
# depth 3 right <numpy_ml.trees.dt.Leaf object at 0x1a3a9f7a20> left_sample 255 right_sample 4
# depth 2 left <numpy_ml.trees.dt.Node object at 0x1a3a9f7ac8> left_sample 259 right_sample 16
# shape: (16, 30)
# ****************************************************************************************************
# before split no_split_data: (16, 30)
# depth 3 split by  (21, 25.89) left train 9 8 right train 7 1
# shape: (9, 30)
# depth 3 left <numpy_ml.trees.dt.Leaf object at 0x1a3a9f7b70> left_sample 9 right_sample 7
# shape: (7, 30)
# depth 3 right <numpy_ml.trees.dt.Leaf object at 0x1a3a9f7be0> left_sample 9 right_sample 7
# depth 2 right <numpy_ml.trees.dt.Node object at 0x1a3a9f7b00> left_sample 259 right_sample 16
# depth 1 left <numpy_ml.trees.dt.Node object at 0x1a0edf1748> left_sample 275 right_sample 180
# shape: (180, 30)
# ****************************************************************************************************
# before split no_split_data: (180, 30)
# depth 2 split by  (27, 0.15075) left train 53 23 right train 127 0
# shape: (53, 30)
# ****************************************************************************************************
# before split no_split_data: (53, 30)
# depth 3 split by  (23, 957.45) left train 28 20 right train 25 3
# shape: (28, 30)
# depth 3 left <numpy_ml.trees.dt.Leaf object at 0x1a3a9f7cc0> left_sample 28 right_sample 25
# shape: (25, 30)
# depth 3 right <numpy_ml.trees.dt.Leaf object at 0x1a3a9f7cf8> left_sample 28 right_sample 25
# depth 2 left <numpy_ml.trees.dt.Node object at 0x1a3a9f7c50> left_sample 53 right_sample 127
# shape: (127, 30)
# depth 2 right <numpy_ml.trees.dt.Leaf object at 0x1a3a9f7d30> left_sample 53 right_sample 127
# depth 1 right <numpy_ml.trees.dt.Node object at 0x1a3a9f7ba8> left_sample 275 right_sample 180




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


