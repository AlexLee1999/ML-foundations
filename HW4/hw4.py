from liblinearpackage.python.liblinearutil import *
import numpy as np

train_file = './hw4_train.dat'
test_file = './hw4_test.dat'
prob_str = '-b 1'

def get_str(s):
    return '-s 0 -c ' + str(1/(2*(10**s))) + ' -e 0.000001 -q'


def main():
    train_data = np.loadtxt(train_file)
    train_y, train_x = train_data[:,-1].tolist(), train_data[:,:-1]
    for i in range(6):
        train_x = np.append(train_x, train_x[:,i:6] * train_x[:,i:i+1], axis=1)
    train_x = np.insert(train_x, 0, 1, axis=1).tolist()
    prob = problem(train_y, train_x)
    
    test_data = np.loadtxt(test_file)
    test_y, test_x = test_data[:,-1].tolist(), test_data[:,:-1]
    for i in range(6):
        test_x = np.append(test_x, test_x[:,i:6] * test_x[:,i:i+1], axis=1)
    test_x = np.insert(test_x, 0, 1, axis=1).tolist()
    C = [-4, -2, 0, 2, 4]
    
    print("Problem 16")
    for c in C:
        print(f"Lambda : {c}")
        param = parameter(get_str(c))
        m = train(prob, param)
        p_label, p_acc, p_val = predict(test_y, test_x, m, prob_str)
        ACC, MSE, SCC = evaluations(test_y, p_label)
    
    print("Problem 17")
    for c in C:
        print(f"Lambda : {c}")
        param = parameter(get_str(c))
        m = train(prob, param)
        p_label, p_acc, p_val = predict(train_y, train_x, m, prob_str)
        ACC, MSE, SCC = evaluations(train_y, p_label)
    
    print("Problem 18")
    train120_y, train120_x = train_y[:120], train_x[:120]
    prob = problem(train120_y, train120_x)
    val_y, val_x = train_y[120:], train_x[120:]
    l = 0
    M = None
    acc = 0
    for c in C:
        print(f"Lambda : {c}")
        param = parameter(get_str(c))
        m = train(prob, param)
        p_label, p_acc, p_val = predict(val_y, val_x, m, prob_str)
        ACC, MSE, SCC = evaluations(val_y, p_label)
        if ACC >= acc:
            l, M, acc = c, m, ACC
    p_label, p_acc, p_val = predict(test_y, test_x, M, prob_str)
    ACC, MSE, SCC = evaluations(test_y, p_label)
    
    print("Problem 19")
    prob = problem(train_y, train_x)
    print(f"Lambda : {l}")
    param = parameter(get_str(l))
    m = train(prob, param)
    p_label, p_acc, p_val = predict(test_y, test_x, m, prob_str)
    ACC, MSE, SCC = evaluations(test_y, p_label)

    print("Problem 20")
    acc = 0
    for c in C:
        aacc = 0
        for i in range(0, 200, 40):
            print(f"Lambda : {c}")
            fold_y, fold_x = train_y[i:i+40], train_x[i:i+40]
            prob = problem(train_y[0:i] + train_y[i+40:], train_x[0:i] + train_x[i+40:])
            param = parameter(get_str(c))
            m = train(prob, param)
            p_label, p_acc, p_val = predict(fold_y, fold_x, m, prob_str)
            ACC, MSE, SCC = evaluations(fold_y, p_label)
            aacc += ACC
        if aacc >= acc:
            acc = aacc
    print(acc/5)

if __name__ == "__main__":
    main()
    