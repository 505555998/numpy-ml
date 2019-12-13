import numpy as np

############################################################
# 1. np.c_ 列合并
X = np.random.randint(0,10,(5,2))
X


np.c_[np.ones(X.shape[0]), X]

# array([[1., 9., 6.],
#        [1., 3., 1.],
#        [1., 9., 5.],
#        [1., 9., 4.],
#        [1., 6., 0.]])



# 2. broadcast
np.atleast_2d(([0.1,0.2,0.3]))
# 同
np.array([[0.1,0.2,0.3]])
np.array([[0.1,0.2,0.3]]).shape # (1, 3)

np.tile([10,100,100],(5,1))
# Out[47]:
# array([[ 10, 100, 100],
#        [ 10, 100, 100],
#        [ 10, 100, 100],
#        [ 10, 100, 100],
#        [ 10, 100, 100]])

np.array([0.1,0.2,0.3,0.4,0.5]) & np.tile([10,100,100],(5,1))
# TypeError: ufunc 'bitwise_and' not supported for the input types,
# and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''


np.dot(np.array([0.1,0.2,0.3,0.4,0.5]),np.tile([10,100,100],(5,1)))


# array([ 15., 150., 150.])



a = np.array([1,2,3]*2,).reshape(2,3)
# array([[1, 2, 3],
#        [1, 2, 3]])

# 矩阵乘法
np.dot(a,a.T)

#  @ 同 matmul。和dot 差别很小，但是matmul 禁止：矩阵和标量乘积
a @ a.T
np.matmul(a,a.T)

# 点乘
np.multiply(a,a)


# 需要broadcast的时候

np.dot(np.array([0.1,0.2,0.3,0.4,0.5]),np.tile([10,100,100],(5,1)))

# (5,) (5,3)  s生成 （3，）



#