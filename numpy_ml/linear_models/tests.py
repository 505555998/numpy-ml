from sklearn.linear_model import LogisticRegression
# 下午测试：

# lr

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
X, y = load_iris(return_X_y=True)

X = X[:100,:]
y = y[:100]

clf = LogisticRegression(penalty = 'l1',
                         fit_intercept=True,
                         C=1.0,
                         random_state=0,
                         solver='saga',
                         multi_class = 'ovr').fit(X, y)
clf.coef_
# array([[ 0.44036482, -0.90696813,  2.30849566,  0.96232763]])
clf.intercept_
# array([-6.61165119])
roc_auc_score(y,clf.predict_proba(X)[:,1])
# 1.0

#############

from numpy_ml.linear_models.lm import *

lr = LogisticRegression(penalty="l1",gamma=1)
lr


lr.fit(X=X,y=y,lr = 0.01,max_iter = 10)
lr.beta
roc_auc_score(y,lr.predict(X))





#############
# change










