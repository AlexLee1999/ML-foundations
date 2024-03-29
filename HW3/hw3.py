import pandas as pd
import numpy as np
import random
import sys
import math
times = 1000


data = []
with open('hw3_train.dat', 'r') as f:
    d = f.readlines()
    for i in d:
        k = i.rstrip().split("\t")
        data.append([i for i in k])

data = np.array(data, dtype=float)
data = np.insert(data, 0, 1.0, axis=1)
data_y = data[:, -1]
data = np.delete(data, -1, axis=1)


def inv(x):
    return np.linalg.pinv(x)


data_inv = inv(data)
w_lin = np.dot(data_inv, data_y)


def cal_err(d, w, y):
    y_hat = np.dot(d, w)
    err = np.subtract(y, y_hat)
    err_t = np.transpose(err)
    err_in = np.dot(err, err_t)
    return err_in

e_wlin = cal_err(data, w_lin, data_y)
print(f"Problem 14 : {e_wlin / 1000}")


def linear_sgd():
    a = 1000000000
    w = np.zeros(shape=(11, 1))
    count = 0
    seedValue = random.randrange(sys.maxsize)
    random.seed(seedValue)
    while a > 1.01 * e_wlin:
        x = random.randint(0, 999)
        x_n = data[x]
        y_n = data_y[x]
        dot = np.dot(np.transpose(w), np.transpose(x_n))
        w = w + 0.002 * (y_n - dot[0]) * np.transpose([x_n])
        a = cal_err(data, np.transpose(w[:, 0]), data_y)
        count += 1
    return count
c_15 = 0
for _ in range(times):
    c_15 += linear_sgd()
print(f"Problem 15 : {c_15 / times}")


def sig(x):
    return 1 / (1 + math.exp(-1 * x))


def err_log(x, w, y):
    dot = np.dot(w, x)
    return np.log(1 + math.exp(-1 * y * dot))


def log_sgd():
    w = np.zeros(shape=(11, 1))

    seedValue = random.randrange(sys.maxsize)
    random.seed(seedValue)
    for _ in range(500):
        x = random.randint(0, 999)
        x_n = data[x]
        y_n = data_y[x]
        dot = np.dot(np.transpose(w), np.transpose(x_n))
        v = -1 * y_n * dot
        w = w + 0.001 * sig(v) * y_n * np.transpose([x_n])
    su = 0
    for i in range(1000):
        x_i = data[i]
        y_i = data_y[i]
        su += err_log(x_i, np.transpose(w[:, 0]), y_i)
    return su / 1000

c_16 = 0
for _ in range(times):
    c_16 += log_sgd()

print(f"Problem 16 : {c_16 / times}")


def log_sgd_with_init():
    w = np.transpose([w_lin])
    seedValue = random.randrange(sys.maxsize)
    random.seed(seedValue)
    for _ in range(500):
        x = random.randint(0, 999)
        x_n = data[x]
        y_n = data_y[x]
        dot = np.dot(np.transpose(w), np.transpose(x_n))
        v = -1 * y_n * dot
        w = w + 0.001 * sig(v) * y_n * np.transpose([x_n])
    su = 0
    for i in range(1000):
        x_i = data[i]
        y_i = data_y[i]
        su += err_log(x_i, np.transpose(w[:, 0]), y_i)
    return su / 1000

c_17 = 0
for _ in range(times):
    c_17 += log_sgd_with_init()
print(f"Problem 17 : {c_17 / times}")

data_test = []
with open('hw3_test.dat', 'r') as f:
    d = f.readlines()
    for i in d:
        k = i.rstrip().split("\t")
        data_test.append([i for i in k])

data_test = np.array(data_test, dtype=float)
data_test = np.insert(data_test, 0, 1.0, axis=1)
data_y_test = data_test[:, -1]
data_test = np.delete(data_test, -1, axis=1)


def sign(x):
    if x > 0:
        return 1
    else:
        return -1


def binary_err(y_hat, y):
    count = 0
    for i in range(len(y_hat)):
        if sign(y_hat[i]) != sign(y[i]):
            count += 1
    return count
in_err = binary_err(np.dot(data, w_lin), data_y) / 1000
out_err = binary_err(np.dot(data_test, w_lin), data_y_test) / 3000
print(f"Problem 18 : {abs(in_err - out_err)}")


def transform_data(d, n, size):
    r_d = np.zeros((size, 10))
    for i in range(size):
        for j in range(1, 11):
            r_d[i, j-1] = math.pow(d[i, j], n)
    return r_d


def merge_transform_data(d, n, size):
    re_d = d
    for i in range(2, n+1):
        r_data = transform_data(d, i, size)
        re_d = np.concatenate((re_d, r_data), axis=1)
    return re_d
data_3 = merge_transform_data(data, 3, 1000)

data_test_3 = merge_transform_data(data_test, 3, 3000)
data_inv_3 = inv(data_3)
w_lin_3 = np.dot(data_inv_3, data_y)
in_err_3 = binary_err(np.dot(data_3, w_lin_3), data_y) / 1000
out_err_3 = binary_err(np.dot(data_test_3, w_lin_3), data_y_test) / 3000
print(f"Problem 19 : {abs(in_err_3 - out_err_3)}")

data_10 = merge_transform_data(data, 10, 1000)
data_test_10 = merge_transform_data(data_test, 10, 3000)

data_inv_10 = inv(data_10)
w_lin_10 = np.dot(data_inv_10, data_y)
in_err_10 = binary_err(np.dot(data_10, w_lin_10), data_y) / 1000
out_err_10 = binary_err(np.dot(data_test_10, w_lin_10), data_y_test) / 3000
print(f"Problem 20 : {abs(in_err_10 - out_err_10)}")

