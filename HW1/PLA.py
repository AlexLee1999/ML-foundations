import random
import numpy as np
SIZE = 11
default_path = './hw1_train.dat'
## read file
def read_file(filename):
    lines = []
    with open(filename, 'r') as file:
        line = file.readline()
        l = line.split()
        for i in range(0, len(l)): 
            l[i] = float(l[i])
        lines.append(l)
        while line:
            line = file.readline()
            if line:
                l = line.split()
                for i in range(0, len(l)): 
                    l[i] = float(l[i])
                lines.append(l)
    return lines

## store x and y in different array
def seperate_x_and_y(lines):
    y_array = []
    for line in lines:
        last = line.pop()
        y_array.append(last)
    return y_array, lines

## add X0
def add_x(x, v):
    for array in x:
        array.insert(0, v)
    return x

## inner product of two vectors
def dot(a, b):
    sum = 0
    for i in range(len(a)):
        sum += a[i]*b[i]
    return sum

## the sign() function
def check_sign(x):
    if x > 0:
        return 1
    else:
        return -1

def check_valid(x, y, w, i):
    a = dot(x[i], w)
    if check_sign(a) != check_sign(y[i]):
        return False
    return True
        
def array_add(a, b):
    c = []
    for i in range(len(a)):
        c.append(a[i]+b[i])
    return c

def array_mul(array, const):
    a = []
    for i in range(len(array)):
        a.append(const*array[i])
    return a

def pla(x, y):
    w = [0] * SIZE
    count = 0
    count_neg = 0
    while count < 500:
        i = random.randint(0,99)
        a = check_valid(x, y, w, i)
        if a == True:
            count +=1
        else:
            tem = array_mul(x[i], y[i])
            w = array_add(w, tem)
            count_neg += 1
            count = 0
    return count_neg, w[0]

def median(l):
    l.sort()
    if len(l) % 2 == 0:
        b = l[int(len(l)/2)]
        c = l[int((len(l)/2) - 1)]
        d = (b + c) / 2
        return d
    if len(l) % 2 > 0:
        return l[int(len(l)/2)]

def scale_down(l, n):
    for num in l:
        num = num/n
    return l

def combined(path, x0_value, n):
    l = []
    w0 = []
    lines = read_file(path)
    y, x = seperate_x_and_y(lines)
    x = add_x(x, x0_value)
    for array in x:
        scale_down(array, n)
    for _ in range(1000):
        a, b = pla(x, y)
        l.append(a)
        w0.append(b)
    return median(l), median(w0)

if __name__ == "__main__":
    median_update_time, median_w0 = combined(default_path, 1, 1)
    print('Problem 16: ' + str(median_update_time))
    print('Problem 17: ' + str(median_w0))
    median_update_time, median_w0 = combined(default_path, 10, 1)
    print('Problem 18: ' + str(median_update_time))
    median_update_time, median_w0 = combined(default_path, 0, 1)
    print('Problem 19: ' + str(median_update_time))
    median_update_time, median_w0 = combined(default_path, 0, 4)
    print('Problem 20: ' + str(median_update_time))

    