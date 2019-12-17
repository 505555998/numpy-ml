# debufg
# todo: 理解 tree 递归return
# 涉及到堆栈 等，执行顺序等，内部的计算机计算机制
# 需要对效率问题进行优化的
# https://www.cnblogs.com/king-lps/p/10748535.html

def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        # 写在上面会一直执行
        a1 = fibonacci(n-1)
        print("a\tn\tn-\ta")
        print("a1",n,n-1,a1)
        a2 = fibonacci(n-2)
        print("a2",n, n - 2,a2)

        return a1+a2

fibonacci(3)
