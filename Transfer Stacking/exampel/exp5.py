# coding: UTF-8
import Transfer_Stacking as TS
import pandas as pd

# same-distribution training data
tarin_data = pd.read_csv('M_Sdata.csv')
# two diff-distribution training data
A1_tarin_data = pd.read_csv('M_Adata1.csv')

# test data
test_data = pd.read_csv('M_Tdata.csv')

Multi_trans_A = {
'trans_A_1' : A1_tarin_data.iloc[:,:-1],
}
Multi_response_A = {
'response_A_1' :  A1_tarin_data.iloc[:,-1] ,
}
trans_S = tarin_data.iloc[:,:-1]
response_S = tarin_data.iloc[:, -1]
test = test_data.iloc[:,:-1]

TS.Transfer_Stacking(trans_S, Multi_trans_A, response_S, Multi_response_A, test,)
