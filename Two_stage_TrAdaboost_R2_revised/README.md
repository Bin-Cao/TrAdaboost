# Two stage TrAdaboost R2 revised

Written using Python, which is suitable for operating systems, e.g., Windows/Linux/MAC OS etc.


## Note
``` javascript
Boosting for Regression Transfer

Please feel free to open issues in the Github : https://github.com/Bin-Cao/TrAdaboost
or 
contact Bin Cao (bcao@shu.edu.cn)
in case of any problems/comments/suggestions in using the code. 

Parameters
----------
trans_S : feature matrix of same-distribution training data.

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
Two_stage_TrAdaboost_R2_rv(trans_S, Multi_trans_A, response_S, Multi_response_A, test, steps_S, N)

References
----------
.. [1] Two_stage_TrAdaboost_R2
Prof. Zhang TY 《 An Introduction to Materials Informatics (II)》

```

## About
Maintained by Bin Cao. Please feel free to open issues in the Github or contact Bin Cao
(bcao@shu.edu.cn) in case of any problems/comments/suggestions in using the code. 

