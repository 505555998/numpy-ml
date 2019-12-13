from sklearn.linear_model import LogisticRegression
import numpy as np
np.random.seed(0)
# 下午测试：
#################################################
#### sklearn

from sklearn.datasets import load_breast_cancer  # 经典二分类
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
X, y = load_breast_cancer(return_X_y=True)



random_order = list(range(len(X)))
np.random.shuffle(random_order)
train_samples = int(len(random_order)*0.8,)
X_train = X[random_order[:train_samples]]
X_test =  X[random_order[train_samples:]]

y_train = y[random_order[:train_samples]]
y_test =  y[random_order[train_samples:]]

########


X.shape
y.shape
#
# set(y) # {0, 1}
#
# clf = LogisticRegression(penalty = 'l1',
#                          fit_intercept=True,
#                          C=1.0,
#                          tol=1e-5,
#                          random_state=0,
#                          solver='saga',
#                          multi_class = 'ovr').fit(X, y)
#
# def eval(X,y):
#     print("coef_",clf.coef_)
#     print("intercept_",clf.intercept_)
#     print(roc_auc_score(y,clf.predict_proba(X)[:,1]))
#
# eval(X_train,y_train)
# # 0.9403484697272122
# eval(X_test,y_test)
# # 0.9297906602254429




####################################################

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


from importlib import reload
from numpy_ml.linear_models import lm
reload(lm)
from numpy_ml.linear_models.lm import *

lr_ml = LogisticRegression(penalty="l1",gamma=10)

lr_ml.fit(X=X_train_scaled,y=y_train,lr = 1e-2,max_iter = 1e3,tol=1e-7,random_state=999 )
lr_ml.beta


roc_auc_score(y_train,lr_ml.predict(X_train_scaled)) # 0.9713697604790419
roc_auc_score(y_test,lr_ml.predict(X_test_scaled)) # 0.96425120


#############
# l1 gamma : 10
# Out[97]:
# array([ 4.86131052e-02,  1.95659294e-05, -5.81226198e-05, -6.41641296e-05,
#        -8.86506261e-04,  2.89806507e-03,  4.58742034e-05, -6.44008467e-03,
#        -9.08706499e-03,  1.10253303e-03,  7.32997757e-03, -7.95324028e-05,
#         2.80317622e-03, -3.21196790e-04,  3.32018981e-05,  1.71673472e-03,
#         7.68302558e-05,  1.24705631e-04,  2.48911031e-05,  5.42402939e-03,
#        -8.87770937e-05, -2.56891359e-03,  2.04310139e-04, -2.76501130e-03,
#        -3.06659914e-03,  2.11931114e-05, -3.75356464e-06, -2.34506019e-03,
#        -6.47021895e-03,  2.89053928e-04,  2.08510688e-04])


# l2 gamma: 10
# array([ 0.16525767, -0.02602156,  0.01471187, -0.02889313, -0.03526415,
#         0.03355124, -0.02608033, -0.05230291, -0.06081399,  0.02833636,
#         0.04848382, -0.02091502,  0.03593375, -0.018863  , -0.02129164,
#         0.03278468,  0.00268435,  0.00139086,  0.00727968,  0.04345536,
#         0.01242952, -0.04147976,  0.01220966, -0.04190486, -0.04139516,
#         0.02182951, -0.0250502 , -0.04027039, -0.05485956,  0.01033543,
#         0.00528756])


