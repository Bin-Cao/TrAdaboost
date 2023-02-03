# TaskTrAdaBoost

Written using Python, which is suitable for operating systems, e.g., Windows/Linux/MAC OS etc.

## Note
``` javascript
Boosting for MultiSource Transfer Learning.

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

gamma : float, for avoiding overfitting 

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

TaskTrAdaBoost(trans_S, Multi_trans_A, label_S, Multi_label_A, test, N, gamma,)

References
----------
.. [1] Yao, Y., & Doretto, G. (2010, June)
Boosting for transfer learning with multiple sources. IEEE.
DOI: 10.1109/CVPR.2010.5539857
```

## About
Maintained by Bin Cao. Please feel free to open issues in the Github or contact Bin Cao
(bcao@shu.edu.cn) in case of any problems/comments/suggestions in using the code. 

