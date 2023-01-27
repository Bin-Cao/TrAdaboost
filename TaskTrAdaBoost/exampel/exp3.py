# coding: UTF-8
import TaskTrAdaBoost as TTB
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
gamma = 0.1

TTB.TaskTrAdaBoost(trans_S, Multi_trans_A, label_S, Multi_label_A, test, N, gamma,)
