# 递归
# https://www.python-course.eu/python3_recursive_functions.php
# #######
# 斐波那契数列

a = []
def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        l = fibonacci(n-1) + fibonacci(n-2)
        a.append((n,n-1,n-2,))
        return l

print([fibonacci(x) for x in range(10)])


##############################
#  构造和tree 一样的格式
# 即两次调用

def fibonacci(n,):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        a1 = fibonacci(n-1)
        print("a1",a1,n-1,)
        a2 = fibonacci(n-2)
        print("a2",a2, n - 2)

        return a1+a2

fibonacci(4)







##############################


def factorial(n):
    print("factorial has been called with n = " + str(n))
    if n == 1:
        return 1
    else:
        res = n * factorial(n-1)
        print("intermediate result for ", n, " * factorial(" ,n-1, "): ",res)
        return res

print(factorial(5))



##############################
# 返回n以内所有数的阶乘：

def fn(n):
    if n == 1:
        return 1
    else:
        return n * fn(n - 1)



print([fn(n) for n in range(1,10)])




##############################
# def tree




def fn(n):
    if n == 1:
        l = 1
    else:
        l =  n * fn(n - 1)
        print(n,l)
    x = 10
    return x



fn(5)
# 2 20
# 3 30
# 4 40
# 5 50


###################################

def fn(n):
    if n == 1:
        l = 1
    else:
        l =  n * fn(n - 1)
        print(n,l)
        print("*"*20)
        l = 2*n * fn(n - 1)
        print("right***",n,l)
    x = 10
    return x

fn(5)


# 2 20
# right*** 2 40
# 3 30
# 2 20
# right*** 2 40
# right*** 3 60
# 4 40
# 2 20
# right*** 2 40
# 3 30
# 2 20
# right*** 2 40
# right*** 3 60
# right*** 4 80
# 5 50
# 2 20
# right*** 2 40
# 3 30
# 2 20
# right*** 2 40
# right*** 3 60
# 4 40
# 2 20
# right*** 2 40
# 3 30
# 2 20
# right*** 2 40
# right*** 3 60
# right*** 4 80
# right*** 5 100
# Out[33]: 10



###############


def fibonacci(n,):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        a1 = fibonacci(n-1)
        print("a1",a1,n-1,)
        a2 = fibonacci(n-2)
        print("a2",a2, n - 2)

        return a1+a2

fibonacci(4)


###############

# 控制次数

def f(n,c=0):
    if c<=10:
        print(n,c)
        return f(n,c = c+1)

f(10)

