# coding: UTF-8
import numpy as np
from sklearn.tree import DecisionTreeRegressor

# =============================================================================
# Public estimators
# =============================================================================

def TrAdaBoost_R2(trans_S, Multi_trans_A, response_S, Multi_response_A, test, N):
    """Boosting for regression transfer. 

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

    response_S : responses of same-distribution training data, real number

    Multi_response_A : dict, responses of diff-distribution training data, real number
    e.g.,
    Multi_response_A = {
    'response_A_1' :  response_1 , 
    'response_A_2' : response_2 ,
    ......
    }

    test : feature matrix of test data

    N: int, the number of estimators in TrAdaBoost_R2

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
    Multi_response_A = {
    'response_A_1' :  A1_tarin_data.iloc[:,-1] , 
    'response_A_2' :  A2_tarin_data.iloc[:,-1] ,
    }

    trans_S = tarin_data.iloc[:,:-1]
    response_S = tarin_data.iloc[:, -1]

    test = test_data.iloc[:,:-1]
    N = 20

    TrAdaBoost_R2(trans_S, Multi_trans_A, response_S, Multi_response_A, test, N)

    References
    ----------
    .. [1] section 4.1
    Pardoe, D., & Stone, P. (2010, June). 
    Boosting for regression transfer. 
    In Proceedings of the 27th International Conference 
    on International Conference on Machine Learning (pp. 863-870).
    """

     # prepare trans_A
    trans_A = list(Multi_trans_A.values())[0]
    if len(Multi_trans_A) == 1:
        pass
    else:
        for i in range(len(Multi_trans_A)-1):
            p = i + 1
            trans_A = np.concatenate((trans_A, list(Multi_trans_A.values())[p]), axis=0)
    # prepare response_A
    response_A = list(Multi_response_A.values())[0]
    if len(Multi_response_A) == 1:
        pass 
    else:
        for i in range(len(Multi_response_A)-1):
            p = i + 1
            response_A = np.concatenate((response_A, list(Multi_response_A.values())[p]), axis=0)
   
    trans_data = np.concatenate((trans_A, trans_S), axis=0)
    trans_response = np.concatenate((response_A, response_S), axis=0)

    row_A = trans_A.shape[0]
    row_S = trans_S.shape[0]
    row_T = test.shape[0]

    if N > row_A:
        print('The maximum of iterations should be smaller than ', row_A)
        
    test_data = np.concatenate((trans_data, test), axis=0)

    # Initialize the weights
    weights_A = np.ones([row_A, 1]) / row_A
    weights_S = np.ones([row_S, 1]) / row_S
    weights = np.concatenate((weights_A, weights_S), axis=0) 

    bata = 1 / (1 + np.sqrt(2 * np.log(row_A / N)))

    # Save prediction response and bata_t
    bata_T = np.zeros([1, N])
    result_response = np.ones([row_A + row_S + row_T, N])

    # Save the prediction response of test data 
    predict = np.zeros([row_T])
    print ('params initial finished.')
    print('='*60)

    trans_data = np.asarray(trans_data, order='C')
    trans_response = np.asarray(trans_response, order='C')
    test_data = np.asarray(test_data, order='C')

    for i in range(N):
        weights = calculate_P(weights)
        result_response[:, i] = base_regressor(trans_data, trans_response, test_data, weights)
        error_rate = calculate_error_rate(response_S, result_response[row_A:row_A + row_S, i],weights[row_A:row_A + row_S, 0])
        # Avoiding overfitting
        if error_rate <= 1e-10 or error_rate > 0.5:
            N = i
            break 
        bata_T[0, i] = error_rate / (1 - error_rate)
        print ('Iter {}-th result :'.format(i))
        print ('error rate :', error_rate, '|| bata_T :', error_rate / (1 - error_rate))
        print('-'*60)

        D_t = np.abs(np.array(result_response[:row_A + row_S, i]) - np.array(trans_response)).max()
        # Changing the data weights of same-distribution training data
        for j in range(row_S):
            weights[row_A + j] = weights[row_A + j] * np.power(bata_T[0, i], -(np.abs(result_response[row_A + j, i] - response_S[j])/D_t))
        # Changing the data weights of diff-distribution training data
        for j in range(row_A):
            weights[j] = weights[j] * np.power(bata, np.abs(result_response[j, i] - response_A[j])/D_t)
    for i in range(row_T):
        predict[i] = np.sum(
            result_response[row_A + row_S + i, int(np.floor(N / 2)):N]) / (N-int(np.floor(N / 2)))
        
    print("TrAdaBoost_R2 is done")
    print('='*60)
    print('The prediction responses of test data are :')
    print(predict)
    return predict

def calculate_P(weights):
    total = np.sum(weights)
    return np.asarray(weights / total, order='C')


def base_regressor(trans_data, trans_response, test_data, weights):
    """
    Base on sampling
    # weight resampling 
    cdf = np.cumsum(weights)
    cdf_ = cdf / cdf[-1]
    uniform_samples = np.random.random_sample(len(trans_data))
    bootstrap_idx = cdf_.searchsorted(uniform_samples, side='right')
    # searchsorted returns a scalar
    bootstrap_idx = np.array(bootstrap_idx, copy=False)
    reg = DecisionTreeRegressor(max_depth=2,splitter='random',max_features="log2",random_state=0)
    reg.fit(trans_data[bootstrap_idx], trans_response[bootstrap_idx])
    """
    reg = DecisionTreeRegressor(max_depth=1,splitter='random',max_features="log2",random_state=0)
    reg.fit(trans_data, trans_response,sample_weight=weights[:,0])
    return reg.predict(test_data)


def calculate_error_rate(response_R, response_H, weight):
    total = np.abs(response_R - response_H).max()
    return np.sum(weight[:] * np.abs(response_R - response_H) / total)