from liblinearpackage.python.liblinearutil import *
import numpy as np

train_file = './hw4_train.dat'
test_file = './hw4_test.dat'
prob_str = '-b 1'
lambda_set = [-4, -2, 0, 2, 4]

def get_str(s):
    return '-s 0 -c ' + str(1/(2*(10**s))) + ' -e 0.000001 -q'


def prepare_file(file):
    data = np.loadtxt(file)
    y, x = data[:,-1].tolist(), data[:,:-1]
    for i in range(6):
        x = np.append(x, x[:,i:6] * x[:,i:i+1], axis=1)
    x = np.insert(x, 0, 1, axis=1).tolist()
    return x, y

def solve(l, problem, x, y):
    p = parameter(get_str(l))
    m = train(problem, p)
    p_L, _, _ = predict(y, x, m, prob_str)
    accuracy, _, _ = evaluations(y, p_L)
    return accuracy, m

def main():
    x_lst, y_lst = prepare_file(train_file)
    test_x_lst, test_y_lst = prepare_file(test_file)
    problem_16_17_19 = problem(y_lst, x_lst)
    print("Problem 16")
    for c in lambda_set:
        print(f"Lambda : {c}")
        solve(c, problem_16_17_19, test_x_lst, test_y_lst)
    
    print("Problem 17")
    for c in lambda_set:
        print(f"Lambda : {c}")
        solve(c, problem_16_17_19, x_lst, y_lst)
    
    print("Problem 18")
    train_y_lst_120, train_x_lst_120 = y_lst[:120], x_lst[:120]
    problem_18 = problem(train_y_lst_120, train_x_lst_120)
    validation_y_lst, validation_x_lst = y_lst[120:], x_lst[120:]
    c_min, acc_min, model_min = 0, 0, None
    for c in lambda_set:
        print(f"Lambda : {c}")
        acc, m_18 = solve(c, problem_18, validation_x_lst, validation_y_lst)
        if acc >= acc_min:
            c_min, model_min, acc_min = c, m_18, acc
    p_label, _, _ = predict(test_y_lst, test_x_lst, model_min, prob_str)
    
    print("Problem 19")
    print(f"Lambda : {c_min}")
    solve(c_min, problem_16_17_19, test_x_lst, test_y_lst)

    print("Problem 20")
    s = 0
    for c in lambda_set:
        count = 0
        for i in range(5):
            print(f"Lambda : {c}, Fold : {i+1}")
            fold_y_lst, fold_x_lst = y_lst[i*40:(i+1)*40], x_lst[i*40:(i+1)*40]
            problem_20 = problem(y_lst[0:i*40] + y_lst[(i+1)*40:], x_lst[0:i*40] + x_lst[(i+1)*40:])
            acc, _ = solve(c, problem_20, fold_x_lst, fold_y_lst)
            count += acc
        if count >= s:
            s = count
    print(f"Best accuracy : {s/5}")

if __name__ == "__main__":
    main()
    