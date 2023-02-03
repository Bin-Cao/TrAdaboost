# coding: UTF-8
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# =============================================================================
# Public estimators
# =============================================================================


def Transfer_Stacking(trans_S, Multi_trans_A, response_S, Multi_response_A, test,):
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
 
    Transfer_Stacking(trans_S, Multi_trans_A, response_S, Multi_response_A, test,)

    References
    ----------
    .. [1] Pardoe, D., & Stone, P. (2010, June). 
    Boosting for regression transfer. 
    In Proceedings of the 27th International Conference 
    on International Conference on Machine Learning (pp. 863-870).

    """
    # generate a pool of experts according the diff-dis datasets
    weak_classifiers_set = []
    reg = DecisionTreeRegressor(max_depth=2,splitter='random',max_features="log2",random_state=0)
    for source in range(len(Multi_trans_A)):
        trans_A = list(Multi_trans_A.values())[source]
        response_A = list(Multi_response_A.values())[source]

        trans_A = np.asarray(trans_A, order='C')
        response_A = np.asarray(response_A, order='C')

        weak_classifier = reg.fit(trans_A, response_A, )
        weak_classifiers_set.append(weak_classifier)
    print('A set of experts is initilized and contains {} classifier'.format(len(weak_classifiers_set)))
    print('='*60)

    row_S = trans_S.shape[0]
    row_T = test.shape[0]
    print ('params initial finished.')

    X = np.array(trans_S)
    Y = np.array(response_S)
    LOOCV_LS_matrix = np.ones([row_S, len(weak_classifiers_set)+1])
    LOOCV_LS_matrix[:,-1] = LOOCV_output(X,Y)
    for j in range(len(weak_classifiers_set)):
        LOOCV_LS_matrix[:,j] = weak_classifiers_set[j].predict(X)
    
    # find the linear combination of hypotheses that minimizes squared error.
    reg = LinearRegression().fit(LOOCV_LS_matrix, Y)
    print('The linear combination of hypotheses is founded:')
    print('coef:', reg.coef_ ,'|| intercept :', reg.intercept_)
    coef = reg.coef_
    intercept = reg.intercept_
    # add the newly clf into the set
    weak_classifiers_set.append(reg.fit(X, Y))

    # save the prediction results of weak classifiers
    result_response = np.ones([row_T, len(weak_classifiers_set)])
    for item in range(len(weak_classifiers_set)):
        result_response[:,item] = weak_classifiers_set[item].predict(np.array(test))
    predict = np.ones(row_T) * intercept
    for j in range(len(coef)):
        predict += coef[j] * result_response[:,j]
    print('ExpBoost is done')
    print('='*60)
    print('The prediction responses of test data are :')
    print(predict)
    return predict


def LOOCV_output(X,Y):
    loo = LeaveOneOut()
    reg = DecisionTreeRegressor(max_depth=2,splitter='random',max_features="log2",random_state=0)
    y_pre_loocv = []
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, _ = Y[train_index], Y[test_index]
        weak_classifier_new = reg.fit(X_train, y_train)
        y_pre = weak_classifier_new.predict(X_test)
        y_pre_loocv.append(y_pre)
    return y_pre_loocv