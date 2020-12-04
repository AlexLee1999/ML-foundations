from liblinearpackage.python.liblinearutil import *
import numpy as np

train_file = './hw4_train.dat'
test_file = './hw4_test.dat'
prob_str = '-b 1'
lambda_set = [-4, -2, 0, 2, 4]

def get_str(s):
    return '-s 0 -c ' + str(1/(2*(10**s))) + ' -e 0.000001 -q'

def main():
    train_data = np.loadtxt(train_file)
    test_data = np.loadtxt(test_file)
    y_lst, x_lst = train_data[:,-1].tolist(), train_data[:,:-1]
    for i in range(6):
        x_lst = np.append(x_lst, x_lst[:,i:6] * x_lst[:,i:i+1], axis=1)
    x_lst = np.insert(x_lst, 0, 1, axis=1).tolist()
    test_y_lst, test_x_lst = test_data[:,-1].tolist(), test_data[:,:-1]
    for i in range(6):
        test_x_lst = np.append(test_x_lst, test_x_lst[:,i:6] * test_x_lst[:,i:i+1], axis=1)
    test_x_lst = np.insert(test_x_lst, 0, 1, axis=1).tolist()
    
    problem_16_17_19 = problem(y_lst, x_lst)
    print("Problem 16")
    for c in lambda_set:
        print(f"Lambda : {c}")
        p_16 = parameter(get_str(c))
        m_16 = train(problem_16_17_19, p_16)
        predict(test_y_lst, test_x_lst, m_16, prob_str)
    
    print("Problem 17")
    for c in lambda_set:
        print(f"Lambda : {c}")
        p_17 = parameter(get_str(c))
        m_17 = train(problem_16_17_19, p_17)
        predict(y_lst, x_lst, m_17, prob_str)
    
    print("Problem 18")
    train_y_lst_120, train_x_lst_120 = y_lst[:120], x_lst[:120]
    problem_18 = problem(train_y_lst_120, train_x_lst_120)
    validation_y_lst, validation_x_lst = y_lst[120:], x_lst[120:]
    c_min, acc_min, model_min = 0, 0, None
    for c in lambda_set:
        print(f"Lambda : {c}")
        param = parameter(get_str(c))
        m = train(problem_18, param)
        p_label, _, _ = predict(validation_y_lst, validation_x_lst, m, prob_str)
        ACC, _, _ = evaluations(validation_y_lst, p_label)
        if ACC >= acc_min:
            c_min, model_min, acc_min = c, m, ACC
    p_label, _, _ = predict(test_y_lst, test_x_lst, model_min, prob_str)
    ACC, _, _ = evaluations(test_y_lst, p_label)
    
    print("Problem 19")
    print(f"Lambda : {c_min}")
    param = parameter(get_str(c_min))
    #run with full training set
    m = train(problem_16_17_19, param)
    predict(test_y_lst, test_x_lst, m, prob_str)

    print("Problem 20")
    s = 0
    for c in lambda_set:
        count = 0
        for i in range(0, 200, 40):
            print(f"Lambda : {c}, Fold : {int(i/40)+1}")
            fold_y_lst, fold_x_lst = y_lst[i:i+40], x_lst[i:i+40]
            prob = problem(y_lst[0:i] + y_lst[i+40:], x_lst[0:i] + x_lst[i+40:])
            param = parameter(get_str(c))
            m = train(prob, param)
            p_label, _, _ = predict(fold_y_lst, fold_x_lst, m, prob_str)
            ACC, _, _ = evaluations(fold_y_lst, p_label)
            count += ACC
        if count >= s:
            s = count
    print(f"Best accuracy : {s/5}")

if __name__ == "__main__":
    main()
    