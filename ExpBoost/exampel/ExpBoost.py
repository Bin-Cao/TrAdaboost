# coding: UTF-8
import numpy as np
from sklearn import tree

# =============================================================================
# Public estimators
# =============================================================================


def ExpBoost(trans_S, Multi_trans_A, label_S, Multi_label_A, test, N):
    """Boosting Expert Ensembles for Rapid Concept Recall.

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

    ExpBoost(trans_S, Multi_trans_A, label_S, Multi_label_A, test, N,)

    References
    ----------
    .. [1] Rettinger, A., Zinkevich, M., & Bowling, M. (2006, July). 
    Boosting expert ensembles for rapid concept recall. 
    In Proceedings of the National Conference on Artificial Intelligence 
    (Vol. 21, No. 1, p. 464). 
    Menlo Park, CA; Cambridge, MA; London; AAAI Press; MIT Press; 1999.

    """
    # generate a pool of experts according the diff-dis datasets
    weak_classifiers_set = []
    for source in range(len(Multi_trans_A)):
        trans_A = list(Multi_trans_A.values())[source]
        label_A = list(Multi_label_A.values())[source]

        trans_A = np.asarray(trans_A, order='C')
        label_A = np.asarray(label_A, order='C')

        clf = tree.DecisionTreeClassifier(criterion="gini", max_depth = 2,max_features="log2", splitter="random",random_state=0)
        weak_classifier = clf.fit(trans_A, label_A,)
        weak_classifiers_set.append(weak_classifier)
    print('A pool of experts is initilized and contains {} classifier'.format(len(weak_classifiers_set)))
    print('='*60)
    
    row_S = trans_S.shape[0]
    row_T = test.shape[0]
    
    test_data = np.concatenate((trans_S, test), axis=0)
    test_data = np.asarray(test_data, order='C')
    
    # initial weight
    weights_S = np.ones([row_S, 1]) / row_S
    predict = np.zeros([row_T])
    alpha_I = np.zeros([1, N])
    result_label = np.ones([row_S + row_T, N])
    print ('params initial finished.')

    for k in range(N):
        weights_S = calculate_ratio_weight(weights_S)
        error_rate_set = []
        clf_new = tree.DecisionTreeClassifier(criterion="gini", max_depth = 2,max_features="log2", splitter="random",random_state=0)
        weak_classifier_new = clf_new.fit(np.array(trans_S), np.array(label_S), sample_weight = weights_S[:, 0])
        weak_classifiers_set.append(weak_classifier_new)
        # save the prediction results of weak classifiers
        _result_label = np.ones([row_S + row_T, len(weak_classifiers_set)])
        for item in range(len(weak_classifiers_set)):
            _result_label[:,item] = weak_classifiers_set[item].predict(test_data)
            _error = calculate_error_rate(label_S, _result_label[0:row_S, item], weights_S)
            error_rate_set.append(_error)
        error_rate_set = np.array(error_rate_set)
        # choise the best weak classifier and remove it from the set 
        classifier_index = np.random.choice(np.flatnonzero(error_rate_set == error_rate_set.min()))
        result_label[:,k] = _result_label[:,classifier_index]
        error = error_rate_set[classifier_index]
        if error > 0.5 or error < 1e-10: break
        alpha_I[0, k] = 0.5 * np.log((1 - error) / error) 
       
        
       # Changing the data weights of same-distribution training data
        for j in range(row_S):
            weights_S[j] = weights_S[j] * np.exp(- alpha_I[0, k] * result_label[j, k] * label_S[j])
        
        print('Iter {}-th result :'.format(k))
        print('error rate :', error, '|| alpha_I :', 0.5 * np.log((1 - error) / error) )
        print('-'*60)
    
    for i in range(row_T):
        res_ = np.sum(result_label[row_S + i, :] * alpha_I[0, :])
        if res_ >= 0:
            predict[i] = 1
        else:
            predict[i] = -1
    print('ExpBoost is done')
    print('='*60)
    print('The prediction labels of test data are :')
    print(predict)
    return predict

def calculate_ratio_weight(weights):
    total = np.sum(weights)
    return np.asarray(weights / total, order='C')

def calculate_error_rate(label_R, label_P, weight):
    total = np.sum(weight)
    return np.sum(weight[:, 0] / total * sign(label_R, label_P))

def sign(label_R, label_P):
    _res = label_R - label_P 
    for j in range(len(label_R)):
        if _res[j] != 0:
            _res[j]=1
    return _res