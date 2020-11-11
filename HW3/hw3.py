import pandas as pd
import numpy as np
import random
import sys
import math

df = pd.read_csv('hw3_train.dat', sep="\t", header=None)
df.insert(0, -1, 1)
df_y = df[10]
df.drop(columns=10, axis=1, inplace=True)
df.rename(columns={-1: 0, 0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10}, inplace=True)
df_3 = df.copy()
df_10 = df.copy()
df_inv = pd.DataFrame(np.linalg.pinv(df.values), df.columns, df.index)
w_lin = df_inv @ df_y

def cal_err(d, w, y):
    y_hat = d @ w
    err = y - y_hat
    err_in = err.transpose() @ err
    return err_in
e_wlin = cal_err(df, w_lin, df_y)

print(f"Problem 14 : {e_wlin / 1000}")

def linear_sgd():
    a = 1000000000
    zero_data = np.zeros(shape=(11, 1))
    w = pd.DataFrame(zero_data)
    count = 0
    seedValue = random.randrange(sys.maxsize)
    random.seed(seedValue)
    while a > 1.01 * e_wlin:
        x = random.randint(0,999)
        x_n = df.loc[[x]]
        y_n = df_y.loc[x]
        dot = w.transpose() @ x_n.transpose()
        x_n_arr = x_n.transpose().values
        w = w + 0.002 * (y_n - dot.values)* x_n_arr
        a = cal_err(df, w[0], df_y)
        count += 1
    return count
c_15 = 0
'''
for _ in range(1000):
    c_15 += linear_sgd()
'''
print(f"Problem 15 : {c_15 / 1000}")

def sig(x):
    return 1/(1+math.exp(-1*x))

def err_log(w, x, y):
    dot = (w @ x).values
    return np.log(1 + math.exp(-1*y*dot[0]))

def log_sgd():
    zero_data = np.zeros(shape=(11, 1))
    w = pd.DataFrame(zero_data)
    seedValue = random.randrange(sys.maxsize)
    random.seed(seedValue)
    for _ in range(500):
        x = random.randint(0,999)
        x_n = df.loc[[x]]
        y_n = df_y.loc[x]
        dot = (w.transpose() @ x_n.transpose()).values
        v = -1*y_n*dot
        x_n_arr = x_n.transpose().values
        w = w + 0.001 * sig(v)*y_n*x_n_arr
    su = 0
    for i in range(1000):
        x_i = df.loc[i]
        y_i = df_y.loc[i]
        su += err_log(w.transpose(), x_i.transpose(), y_i)
    return su /1000

c_16 = 0
'''
for _ in range(1000):
    c_16+=log_sgd()
'''
print(f"Problem 16 : {c_16 / 1000}")

def log_sgd_with_init():
    w = pd.DataFrame(w_lin)
    seedValue = random.randrange(sys.maxsize)
    random.seed(seedValue)
    for _ in range(500):
        x = random.randint(0,999)
        x_n = df.loc[[x]]
        y_n = df_y.loc[x]
        dot = (w.transpose() @ x_n.transpose()).values
        v = -1*y_n*dot
        x_n_arr = x_n.transpose().values
        w = w + 0.001 * sig(v)*y_n*x_n_arr
    su = 0
    for i in range(1000):
        x_i = df.loc[i]
        y_i = df_y.loc[i]
        su += err_log(w.transpose(), x_i.transpose(), y_i)
    return su /1000

c_17 = 0
'''
for _ in range(1000):
    c_17 += log_sgd_with_init()
'''
print(f"Problem 17 : {c_17 / 1000}")

df_test = pd.read_csv('hw3_test.dat', sep="\t", header=None)
df_test.insert(0, -1, 1)
df_y_test = df_test[10]
df_test.drop(columns=10, axis=1, inplace=True)
df_test.rename(columns={-1: 0, 0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10}, inplace=True)
def sign(x):
    if x > 0:
        return 1
    else:
        return -1

def binary_err(y_hat, y):
    count =0
    for i in range(len(y_hat)):
        if sign(y_hat[i]) != sign(y[i]):
            count +=1
    return count
print(f"Problem 18 : {abs(binary_err(df @ w_lin, df_y)/1000 - binary_err(df_test @ w_lin, df_y_test)/3000)}")

def transform_data(d, n):
    for i in d:
        for j in d[i]:
            d.replace(j, j**n, inplace=True)
    return d

transform_data(df_3, 3)
df_inv_3 = pd.DataFrame(np.linalg.pinv(df_3.values), df_3.columns, df_3.index)
w_lin_3 = df_inv_3 @ df_y
print(f"Problem 19 : {abs(cal_err(df_3, w_lin_3, df_y)/1000 - cal_err(df_test, w_lin_3, df_y_test)/3000)}")
transform_data(df_10, 10)
print(df_10)
df_inv_10 = pd.DataFrame(np.linalg.pinv(df_10.values), df_10.columns, df_10.index)
w_lin_10 = df_inv_10 @ df_y
print(f"Problem 20 : {abs(cal_err(df_10, w_lin_10, df_y)/1000 - cal_err(df_test, w_lin_10, df_y_test)/3000)}")

