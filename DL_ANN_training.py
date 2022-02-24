# ====================================================================================================
# ANN training
# ====================================================================================================
# import library

import numpy as np
# ----------------------------------------------------------------------------------------------------
# recursive algorithm

n = 5
m = 1

for i in range(n):  # 0 ~ 4

    m = m*(i+1)


def factorial(n):

    if n == 1:
        return 1

    else:
        return n*factorial(n-1)
# ----------------------------------------------------------------------------------------------------
# dynamic programming

# naive Fibonacci -> more time


def fibonacci(n):

    if n <= 2:
        return 1

    else:
        return fibonacci(n-1) + fibonacci(n-2)

# memorized DP Fibonacci -> less time


def m_fibonacci(n):

    global memo

    if memo[n-1] != 0:
        return memo[n-1]

    elif n <= 2:
        memo[n-1] = 1
        return memo[n-1]

    else:
        memo[n-1] = m_fibonacci(n-1) + m_fibonacci(n-2)
        return memo[n-1]


n = 10
memo = np.zeros(n)

m_fibonacci(n)
# ----------------------------------------------------------------------------------------------------
# time

# naive
# n = 30
# %timeit fibonacci(30)

# memorized DP
# memo = np.zeros(n)
# %timeit m_fibonacci(30)
# ====================================================================================================
