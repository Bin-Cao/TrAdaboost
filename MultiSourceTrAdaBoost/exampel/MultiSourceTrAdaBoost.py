# coding: UTF-8
import numpy as np
import copy
from sklearn import tree

# =============================================================================
# Public estimators
# =============================================================================


def MultiSourceTrAdaBoost(trans_S, Multi_trans_A, label_S, Multi_label_A, test, N):
    """Boosting for MultiSource Transfer Learning.

    Please feel free to open issues in the Github : https://github.com/Bin-Cao/TrAdaboost
    or 
    contact Bin Cao (bcao@shu.edu.cn)
    in case of any problems/comments/suggestions in using the code. 

    Parameters
    ----------
    trans_S : feature matrix of same-distribution training data

    Multi_trans_A : dict, feature matrix of diff-distribution training data
    e.g.,
    Multi_trans_A = {
    'trans_A_1' :  data_1 , 
    'trans_A_2' : data_2 ,
    ......
    }
    data_1 : feature matrix of diff-distribution training dataset 1
    data_2 : feature matrix of diff-distribution training dataset 2

    label_S : label of same-distribution training data, -1 or 1

    Multi_label_A : dict, label of diff-distribution training data, -1 or 1
    e.g.,
    Multi_label_A = {
    'label_A_1' :  label_1 , 
    'label_A_2' : label_2 ,
    ......
    }
    label_1 : label of diff-distribution training dataset 1, -1 or 1
    label_1 : label of diff-distribution training dataset 2, -1 or 1

    test : feature matrix of test data

    N : int, default=20
    the number of weak estimators

    Examples
    --------
    import pandas as pd
    # same-distribution training data
    tarin_data = pd.read_csv('M_Sdata.csv')
    # two diff-distribution training data
    A1_tarin_data = pd.read_csv('M_Adata1.csv')
    A2_tarin_data = pd.read_csv('M_Adata2.csv')
    # test data
    test_data = pd.read_csv('M_Tdata.csv')

    Multi_trans_A = {
    'trans_A_1' : A1_tarin_data.iloc[:,:-1],
    'trans_A_2' : A2_tarin_data.iloc[:,:-1]
    }
    Multi_label_A = {
    'label_A_1' :  A1_tarin_data.iloc[:,-1] , 
    'label_A_2' :  A2_tarin_data.iloc[:,-1] ,
    }
    trans_S = tarin_data.iloc[:,:-1]
    label_S = tarin_data.iloc[:, -1]
    test = test_data.iloc[:,:-1]
    N = 20

    MultiSourceTrAdaBoost(trans_S, Multi_trans_A, label_S, Multi_label_A, test, N)

    References
    ----------
    .. [1] Yao, Y., & Doretto, G. (2010, June)
    Boosting for transfer learning with multiple sources. IEEE.
    DOI: 10.1109/CVPR.2010.5539857

    """
    # prepare trans_A
    trans_A = list(Multi_trans_A.values())[0]
    if len(Multi_trans_A) == 1:
        pass
    else:
        for i in range(len(Multi_trans_A)-1):
            p = i + 1
            trans_A = np.concatenate((trans_A, list(Multi_trans_A.values())[p]), axis=0)
    # prepare label_S
    label_A = list(Multi_label_A.values())[0]
    if len(Multi_label_A) == 1:
        pass 
    else:
        for i in range(len(Multi_label_A)-1):
            p = i + 1
            label_A = np.concatenate((label_A, list(Multi_label_A.values())[p]), axis=0)
   
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
    # one-dim column in the shape of ((row_A+row_S),1), column vector
    weights = np.concatenate((weights_A, weights_S), axis=0) 

    alpha_S = 0.5 * np.log((1 + np.sqrt(2 * np.log(row_A / N))))

    # Save prediction labels and bata_t
    alpha_T = np.zeros([1, N])
    result_label = np.ones([row_A + row_S + row_T, N])
    # output label
    predict = np.zeros([row_T])
    print ('params initial finished.')
    print('='*60)

    trans_data = np.asarray(trans_data, order='C')
    trans_label = np.asarray(trans_label, order='C')
    test_data = np.asarray(test_data, order='C')

    for i in range(N):
        weights = calculate_ratio_weight(weights)

        result_label[:, i], error_rate , Source_index, start = Multi_train_classifier(Multi_trans_A, label_S,trans_data, trans_label, test_data, weights,row_A,row_S)
        # Avoiding overfitting
        if error_rate <= 1e-10:
            N = i
            break  

        alpha_T[0, i] = 0.5 * np.log((1 - error_rate) / error_rate) 
        print('Iter {}-th result :'.format(i))
        print('The {}-th diff-distribution training dataset is chosen to transfer'.format(Source_index))
        print('error rate :', error_rate, '|| alpha_T :', 0.5 * np.log((1 - error_rate) / error_rate) )
        print('-'*60)

        # Changing the data weights of same-distribution training data
        for j in range(row_S):
            weights[row_A + j] = weights[row_A + j] * np.exp(alpha_T[0, i] * np.abs(result_label[row_A + j, i] - label_S[j]))
        # Changing the data weights of diff-distribution training data
        for j in range( len(list(Multi_trans_A.values())[Source_index]) ):
            loc = start + j
            weights[loc] = weights[loc] * np.exp(-alpha_S * np.abs(result_label[loc, i] - label_A[loc]))
    
    for i in range(row_T):
        res_ = np.sum(result_label[row_A + row_S + i, :] * alpha_T[0, :])
        if res_ >= 0:
            predict[i] = 1
        else:
            predict[i] = -1
    print("MultiSourceTrAdaBoost is done")
    print('='*60)
    print('The prediction labels of test data are :')
    print(predict)
    return predict


def calculate_ratio_weight(weights):
    total = np.sum(weights)
    return np.asarray(weights / total, order='C')


def train_classifier(trans_data, trans_label, test_data, ratio_weight):
    clf = tree.DecisionTreeClassifier(criterion="gini", max_depth = 2, max_features="log2", splitter="best",random_state=0)
    clf.fit(trans_data, trans_label, sample_weight=ratio_weight[:, 0])
    return clf.predict(test_data)

def Multi_train_classifier(Multi_trans_A,label_S, trans_data, trans_label, test_data, weights,row_A,row_S):
    _result_label = np.ones([len(test_data), len(Multi_trans_A)])
    error_record = []
    start_record = []
    start = 0
    for item in range(len(Multi_trans_A)):
        start_record.append(start)
        sub_dataset = list(Multi_trans_A.values())[item]
        data_dim = len(sub_dataset)
        # train a classifier with the 'item'-th data source
        _trans_data = np.concatenate((trans_data[start : start + data_dim], trans_data[row_A:row_A + row_S]), axis=0) 
        _trans_label = np.concatenate((trans_label[start : start + data_dim], trans_label[row_A:row_A + row_S]), axis=0) 
        _ratio_weight = np.concatenate((weights[start : start + data_dim], weights[row_A:row_A + row_S]), axis=0) 
        _result_label[:, item] = train_classifier(_trans_data, _trans_label, test_data, _ratio_weight)
        start += data_dim
        # cal error rate 
        _error_rate = calculate_error_rate(label_S, _result_label[row_A:row_A + row_S, item],weights[row_A:row_A + row_S, :])
        if _error_rate > 0.5:
            _error_rate = 1 - _error_rate 
            # for a binary classifier 
            # reverse the prediction label -1 to 1; 1 to -1.
            pre_labels = copy.deepcopy(_result_label[:, item])
            _result_label[:, item] = -pre_labels     
        error_record.append(_error_rate)
    error_record = np.array(error_record)
    # choise the best classifier
    classifier_index = np.random.choice(np.flatnonzero(error_record == error_record.min()))
    return _result_label[:,classifier_index], error_record[classifier_index], classifier_index,start_record[classifier_index]

def calculate_error_rate(label_R, label_P, weight):
    total = np.sum(weight)
    return np.sum(weight[:, 0] / total * sign(label_R, label_P))

def sign(label_R, label_P):
    _res = label_R - label_P 
    for j in range(len(label_R)):
        if _res[j] != 0:
            _res[j]=1
    return _res