# 斐波那契数列


def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

print([fibonacci(x) for x in range(10)])




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
    return l



def fin_fn(n):
    l = fn(n)
    print(n,l)
    x = 2
    return 2


fin_fn(5)


fn(5)



print([fn(n) for n in range(1,10)])
