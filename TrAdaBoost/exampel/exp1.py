# coding: UTF-8
import TrAdaBoost as TB
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
pre, _, _ = TB.TrAdaBoost(trans_S, trans_A, label_S, label_A, trans_S, 8)
"""

# example of book
# [an introduction of materials informatics II, Tong-yi Zhang]
pre_err = []
for i in range(20):
    N = i+1
    pre, _, _ = TB.TrAdaBoost(trans_S, trans_A, label_S, label_A, trans_S, N)
    pre_err.append(sum(abs(pre - label_S))/len(trans_S))

_,error,misclassify_list = TB.TrAdaBoost(trans_S, trans_A, label_S, label_A, test, N=20)

fig = plt.figure(figsize=(7,5))
ax1 = fig.add_subplot(111)
ax1.plot(range(6,20),misclassify_list[5:19],'o-',color="red",label ='the N-th classifier')
ax1.plot(range(6,20),pre_err[5:19],'o-',color="k",label ='TrAdaBoost')
ax1.set_ylabel('error rate')
ax1.set_xlabel('iterations')


ax2 = ax1.twinx()
ax2.plot(range(6,20),error[5:19],'o-',color="b",label='weighted error rate')
ax2.set_ylabel('weighted error rate')
ax2.set_xlabel('Same')


plt.xticks(range(6,20))
plt.yticks(np.linspace(0,0.16,9))
plt.grid()
ax1.legend(loc=5)

#ax2.legend(loc=3)
plt.savefig('iteration number.png',bbox_inches = 'tight',dpi=600)
plt.show()

"""