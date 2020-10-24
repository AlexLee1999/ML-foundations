import random

INF = 1000000000
class Data:
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def __str__(self):
        return(f"x: {self._x}, y: {self._y}")
    
    def __lt__(self, other):
         return self._x < other._x

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y

def print_data_list(lst):
    for element in lst:
        print(element)

def gen_x():
    return random.uniform(-1,1)

def gen_one_data(prob):
    x = gen_x()
    y = sign(x)
    f = random.random()
    if f < prob:
        y *= (-1)
    return x, y

def gen_data(size, prob):
    data_lst = []
    for _ in range(size):
        x, y = gen_one_data(prob)
        new_data = Data(x, y)
        data_lst.append(new_data)
    return data_lst

def sign(x):
    if x > 0:
        return 1
    else:
        return -1

def calculate_ein(s, theta, lst):
    count = 0
    for element in lst:
        x = element.get_x()
        y = element.get_y()
        if (s * sign(x - theta)) != y:
            count += 1
    return count / len(lst)

def calculate_eout(s, theta, prob):
    if s == -1:
        return 1 - (abs(theta) / 2)
    else:
        return abs(theta) / 2

        


def gen_theta_lst(lst):
    re_lst = []
    re_lst.append(-1)
    for i in range(len(lst)-1):
        if lst[i].get_x() != lst[i + 1].get_x():
            tem = (lst[i].get_x() + lst[i + 1].get_x()) / 2
            re_lst.append(tem)
    return re_lst


def decision_stump(lst):
    lst.sort()
    theta_lst = gen_theta_lst(lst)
    min_num = INF
    t = 0
    s = 0
    for theta in theta_lst:
        s_p = calculate_ein(1, theta, lst)
        s_n = calculate_ein(-1, theta, lst)
        if s_n >= s_p:
            if s_p < min_num:
                min_num = s_p
                s = 1
                t = theta
        else:
            if s_n < min_num:
                min_num = s_n
                s = -1
                t = theta
    return min_num, t, s
        
if __name__ == "__main__":
    sum_16 = 0
    for _ in range(10000):
        lst = gen_data(2, 0)
        ein, theta, s = decision_stump(lst)
        eout = calculate_eout(s, theta, 0)
        n = eout - ein 
        sum_16 += n
    print(f"problem 16: {sum_16/10000}")

    sum_17 = 0
    for _ in range(10000):
        lst = gen_data(20, 0)
        ein, theta, s = decision_stump(lst)
        eout = calculate_eout(s, theta, 0.1)
        n = eout - ein 
        sum_17 += n
    print(f"problem 17: {sum_17/10000}")

    sum_18 = 0
    for _ in range(10000):
        lst = gen_data(2, 0.1)
        ein, theta, s = decision_stump(lst)
        eout = calculate_eout(s, theta, 0)
        n = eout - ein 
        sum_18 += n
    print(f"problem 18: {sum_18/10000}")

    sum_19 = 0
    for _ in range(10000):
        lst = gen_data(20, 0.1)
        ein, theta, s = decision_stump(lst)
        eout = calculate_eout(s, theta, 0.1)
        n = eout - ein 
        sum_19 += n
    print(f"problem 19: {sum_19/10000}")

    sum_20 = 0
    for _ in range(10000):
        lst = gen_data(200, 0.1)
        ein, theta, s = decision_stump(lst)
        eout = calculate_eout(s, theta, 0.1)
        n = eout - ein 
        sum_20 += n
    print(f"problem 20: {sum_20/10000}")

