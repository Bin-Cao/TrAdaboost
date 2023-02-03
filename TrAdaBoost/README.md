# TrAdaBoost

Written using Python, which is suitable for operating systems, e.g., Windows/Linux/MAC OS etc.

## Note
``` javascript
Boosting for Transfer Learning.

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
```

## About
Maintained by Bin Cao. Please feel free to open issues in the Github or contact Bin Cao
(bcao@shu.edu.cn) in case of any problems/comments/suggestions in using the code. 

