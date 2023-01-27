# coding: UTF-8
import numpy as np
import copy
from sklearn import tree

# =============================================================================
# Public estimators
# =============================================================================

def TrAdaBoost(trans_S, trans_A, label_S, label_A, test, N = 20):
    """Boosting for Transfer Learning.

    Please feel free to open issues in the Github : https://github.com/Bin-Cao/TrAdaboost
    or 
    contact Bin Cao (bcao@shu.edu.cn)
    in case of any problems/comments/suggestions in using the code. 

    Parameters
    ----------
    trans_S : feature matrix of same-distribution training data

    trans_A : feature matrix of diff-distribution training data

    label_S : label of same-distribution training data, 0 or 1

    label_A : label of diff-distribution training data, 0 or 1

    test : feature matrix of test data

    N : int, default=20
    the number of weak estimators

    Examples
    --------
    import pandas as pd
    # same-distribution training data
    tarin_data = pd.read_csv('Sdata.csv')
    # diff-distribution training data
    A_tarin_data = pd.read_csv('Adata.csv')
    # test data
    test_data = pd.read_csv('Tdata.csv')

    trans_S = tarin_data.iloc[:,:-1]
    trans_A = A_tarin_data.iloc[:,:-1]
    label_S = tarin_data.iloc[:, -1]
    label_A = A_tarin_data.iloc[:,-1]
    test = test_data.iloc[:,:-1]
    N = 10

    TrAdaBoost(trans_S, trans_A, label_S, label_A, test, N)

    References
    ----------
    .. [1] Dai, W., Yang, Q., et al. (2007). 
    Boosting for Transfer Learning.(2007), 193--200. 
    In Proceedings of the 24th international conference on Machine learning.

    .. [2] GitHub: https://github.com/chenchiwei/tradaboost/blob/master/TrAdaboost.py
    """

    trans_data = np.concatenate((trans_A, trans_S), axis=0)
    trans_label = np.concatenate((label_A, label_S), axis=0)

    row_A = trans_A.shape[0]
    row_S = trans_S.shape[0]
    row_T = test.shape[0]

    if N >= row_A:
        print('The maximum of iterations should be smaller than ', row_A)

    test_data = np.concatenate((trans_data, test), axis=0)

    # Initialize the weights
    weights_A = np.ones([row_A, 1]) / row_A
    weights_S = np.ones([row_S, 1]) / row_S
    weights = np.concatenate((weights_A, weights_S), axis=0) 

    bata = 1 / (1 + np.sqrt(2 * np.log(row_A / N)))

    # Save prediction labels and bata_t
    bata_T = np.zeros([1, N])
    result_label = np.ones([row_A + row_S + row_T, N])

    # Save the prediction labels of test data 
    predict = np.zeros([row_T])
    print ('params initial finished.')
    print('='*60)

    trans_data = np.asarray(trans_data, order='C')
    trans_label = np.asarray(trans_label, order='C')
    test_data = np.asarray(test_data, order='C')

    for i in range(N):
        P = calculate_P(weights)
        result_label[:, i] = train_classify(trans_data, trans_label, test_data, P)
        error_rate = calculate_error_rate(label_S, result_label[row_A:row_A + row_S, i],weights[row_A:row_A + row_S, :])
        if error_rate > 0.5:
            error_rate = 1 - error_rate 
            # for a binary classifier 
            # reverse the prediction label 0 to 1; 1 to 0.
            pre_labels = copy.deepcopy(result_label[:, i])
            result_label[:, i] = np.invert(pre_labels.astype(np.int32)) + 2

        # Avoiding overfitting
        elif error_rate <= 1e-10:
            N = i
            break 
        bata_T[0, i] = error_rate / (1 - error_rate)
        print ('Iter {}-th result :'.format(i))
        print ('error rate :', error_rate, '|| bata_T :', error_rate / (1 - error_rate))
        print('-'*60)

        # Changing the data weights of same-distribution training data
        for j in range(row_S):
            weights[row_A + j] = weights[row_A + j] * np.power(bata_T[0, i], (-np.abs(result_label[row_A + j, i] - label_S[j])))
        # Changing the data weights of diff-distribution training data
        for j in range(row_A):
            weights[j] = weights[j] * np.power(bata, np.abs(result_label[j, i] - label_A[j]))
    
    for i in range(row_T):
        left = np.sum(
            result_label[row_A + row_S + i, int(np.ceil(N / 2)):N] * np.log(1 / bata_T[0, int(np.ceil(N / 2)):N]))
        right = 0.5 * np.sum(np.log(1 / bata_T[0, int(np.ceil(N / 2)):N]))
        if left >= right:
            predict[i] = 1
        else:
            predict[i] = 0
    print("TrAdaBoost is done")
    print('='*60)
    print('The prediction labels of test data are :')
    print(predict)
    return predict

def calculate_P(weights):
    total = np.sum(weights)
    return np.asarray(weights / total, order='C')

def train_classify(trans_data, trans_label, test_data, P):
    clf = tree.DecisionTreeClassifier(criterion="gini", max_depth = 3,max_features="log2", splitter="best",random_state=0)
    clf.fit(trans_data, trans_label, sample_weight=P[:, 0])
    return clf.predict(test_data)

def calculate_error_rate(label_R, label_H, weight):
    total = np.sum(weight)
    return np.sum(weight[:, 0] / total * np.abs(label_R - label_H))