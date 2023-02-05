# coding: UTF-8
import TrAdaBoost as TB
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
N = 4

TB.TrAdaBoost(trans_S, trans_A, label_S, label_A, test, N)