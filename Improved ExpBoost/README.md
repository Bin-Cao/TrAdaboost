# ExpBoost_R (improved)

We appley the strategy of phase I in TaskTrAdaBoost to generate the expert pool for providing more experts based on diff-distribution datasets (concepts)


Written using Python, which is suitable for operating systems, e.g., Windows/Linux/MAC OS etc.


## Note
``` javascript
Boosting Expert Ensembles for Rapid Concept Recall.

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
```

## About
Maintained by Bin Cao. Please feel free to open issues in the Github or contact Bin Cao
(bcao@shu.edu.cn) in case of any problems/comments/suggestions in using the code. 

