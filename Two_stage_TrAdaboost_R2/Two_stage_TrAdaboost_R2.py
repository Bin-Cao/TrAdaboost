# coding: UTF-8
import copy
import numpy as np
import AdaBoost_R2_T as Regmodel
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut
from sklearn.tree import DecisionTreeRegressor

# =============================================================================
# Public estimators
# =============================================================================


def Two_stage_TrAdaboost_R2(trans_S, Multi_trans_A, response_S, Multi_response_A, test, steps_S, N):
    """Boosting for Regression Transfer

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

    steps_S: int, the number of steps (see Algorithm 3)

    N: int, the number of estimators in AdaBoost_R2_T

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
    steps_S = 10
    Two_stage_TrAdaboost_R2(trans_S, Multi_trans_A, response_S, Multi_response_A, test, steps_S, N)

    References
    ----------
    .. [1] Algorithm 3 
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

    # Initialize the weights
    weight  = np.ones(row_A+row_S)/(row_A+row_S)
    bata_T = np.zeros(steps_S)
    
    print ('params initial finished.')
    print('='*60)

    # generate a pool of AdaBoost_R2_T
    AdaBoost_pre = []
    model_error = []
    for i in range(steps_S):
        res_ = Regmodel.AdaBoost_R2_T(trans_data, trans_response, test, weight,row_A, N )
        AdaBoost_pre.append(res_)
        LOOCV_MSE = LOOCV_test(trans_data, trans_response,  weight,row_A, N)
        model_error.append(LOOCV_MSE)

        """
        # update the data weights
        # weight resampling 
        cdf = np.cumsum(weight)
        cdf_ = cdf / cdf[-1]
        uniform_samples = np.random.random_sample(len(trans_data))
        bootstrap_idx = cdf_.searchsorted(uniform_samples, side='right')
        # searchsorted returns a scalar
        bootstrap_idx = np.array(bootstrap_idx, copy=False)
        reg = DecisionTreeRegressor(max_depth=2,splitter='random',max_features="log2",random_state=0)
        reg.fit(trans_data[bootstrap_idx], trans_response[bootstrap_idx])
        pre_res = reg.predict(trans_data)
        E_t = calculate_error_rate(trans_response, pre_res, weight)
        """
        # In order to ensure that the results are not random,
        # the weights are adjusted by the built-in method 
        reg = DecisionTreeRegressor(max_depth=2,splitter='random',max_features="log2",random_state=0)
        reg.fit(trans_data, trans_response,sample_weight = weight)
        pre_res = reg.predict(trans_data)
        E_t = calculate_error_rate(trans_response, pre_res, weight)

        bata_T[i] =  E_t / (1 - E_t)

        # Changing the data weights of same-distribution training data
        total_w_S = row_S/(row_A+row_S) + i/(steps_S-1)*(1 - row_S/(row_A+row_S))
        weight[row_A : row_A+row_S] =  (weight[row_A : row_A+row_S] / weight[row_A : row_A+row_S].sum()) * total_w_S
        # Changing the data weights of diff-distribution training data
        """
        # for saving computation power, we apply the strategy in MultiSourceTrAdaBoost to update the weights
        # see: 10.1109/CVPR.2010.5539857
        for j in range(row_A):
            weight[j] = weight[j] * np.exp(-bata_T[i] * np.abs(trans_response[j] - pre_res[j]))
        weight[0:row_A] =  weight[0:row_A] * (1-total_w_S) / weight[0:row_A].sum()
        """  
        beta_t = binary_search(total_w_S,weight,trans_response,pre_res,row_A,beta_t_range = (0.01,1,0.01),tal=0.05)
        if beta_t == None:
            for j in range(row_A):
                weight[j] = weight[j] * np.exp(-bata_T[i] * np.abs(trans_response[j] - pre_res[j]))
            weight[0:row_A] =  weight[0:row_A] * (1-total_w_S) / weight[0:row_A].sum()
        else:
            D_t = np.abs(trans_response[0:row_A] - pre_res[0:row_A]).max()
            for j in range(row_A):
                weight[j] = weight[j] * np.power(beta_t, np.abs(trans_response[j] - pre_res[j])/D_t)
            weight[0:row_A] =  weight[0:row_A] * (1-total_w_S) / weight[0:row_A].sum()

        print('Iter {}-th result :'.format(i))
        print('{} AdaBoost_R2_T model has been instantiated :'.format(len(model_error)), '|| E_t :', E_t )
        print('beta_t calculated by binary search is : ',beta_t)
        print('-'*60)
      
    model_error = np.array(model_error)
    min_index = np.random.choice(np.flatnonzero(model_error == model_error.min()))
    print('Two_stage_TrAdaboost_R2 is done')
    print('='*60)
    print('The minimum mean square error :',model_error[min_index])
    print('The prediction responses of test data are :')
    print(AdaBoost_pre[min_index])
    return AdaBoost_pre[min_index]


def LOOCV_test(trans_data, trans_response, weight,row_A, N):
    loo = LeaveOneOut()
    X = np.array(trans_data)
    Y = np.array(trans_response)
    y_pre_loocv = []
    cal = 0
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, _ = Y[train_index], Y[test_index]
        w_train, _ = weight[train_index], weight[test_index]
        if cal <= row_A-1:
            y_pre = Regmodel.AdaBoost_R2_T(X_train, y_train, X_test, w_train,row_A-1, N )
        else:
            y_pre = Regmodel.AdaBoost_R2_T(X_train, y_train, X_test, w_train,row_A, N )
        y_pre_loocv.append(y_pre)
    
    return mean_squared_error(trans_response,y_pre_loocv)


def calculate_error_rate(response_R, response_H, weight):
    total = np.abs(response_R - response_H).max()
    return np.sum(weight[:] * np.abs(response_R - response_H) / total)

# binary_search strategy
def binary_search(total_w_S,__weight,trans_response,pre_res,row_A,beta_t_range = (0.01,1,0.01),tal=0.05):
    # beta_t_range is the search range of beta_t, default = (0.01,1,0.01)
    # viz., beta_t is searched in the interval of 0 to 1, with the step of 0.01 by binary_search
    
    D_t = np.abs(trans_response[0:row_A] - pre_res[0:row_A]).max()
    _list = np.arange(beta_t_range[0],beta_t_range[1],beta_t_range[2])
    low = 0          
    high = len(_list)-1
    while low <= high:   
        weight = copy.deepcopy(__weight) 
        mid = int(np.floor((low+high)/2))
        guess = _list[mid]
        # test beta_t
        for j in range(row_A):
            weight[j] = weight[j] * np.power(guess, np.abs(trans_response[j] - pre_res[j])/D_t)
        diff = (1-total_w_S) -  weight[0:row_A].sum()
        if abs(diff) <= tal:     
            return guess
        # exceed the convergence crtiterion
        elif diff > 0:
            low = mid + 1   
        else:  
            high = mid -1               
      
    print("UNABLE TO COVERGEE IN BINARY SEARCHING")
    return None


    
    
    