🤝🤝🤝 Please star ⭐️ it for promoting open source projects 🌍 ! Thanks !

if you have any questions or need help, you are welcome to contact me

If you are using this code, please cite:
+ Cao Bin, Zhang Tong-yi, Xiong Jie, Zhang Qian, Sun Sheng. Package of Boosting-based transfer learning [2023SR0525555], 2023, Software copyright, GitHub : github.com/Bin-Cao/TrAdaboost.
  
## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Bin-Cao/TrAdaboost&type=Date)](https://star-history.com/#Bin-Cao/TrAdaboost&Date)


# TrAdaBoost : Boosting for transfer learning

Transfer learning allows leveraging the knowledge of source domains, available a priori, to help training a classifier for a target domain, where the available data is scarce.


[![Security Status](https://www.murphysec.com/platform3/v3/badge/1626904646967132160.svg)](https://www.murphysec.com/accept?code=645babf2266d3ebb42b1005074b53306&type=1&from=2)



## Models
(1) classification
+ [TrAdaBoost](https://github.com/Bin-Cao/TrAdaboost/blob/main/TrAdaBoost)
+ [MultiSourceTrAdaBoost](https://github.com/Bin-Cao/TrAdaboost/blob/main/MultiSourceTrAdaBoost)
+ [TaskTrAdaBoost](https://github.com/Bin-Cao/TrAdaboost/blob/main/TaskTrAdaBoost)
+ [ExpBoost](https://github.com/Bin-Cao/TrAdaboost/tree/main/ExpBoost)
+ [Improved ExpBoost](https://github.com/Bin-Cao/TrAdaboost/tree/main/Improved%20ExpBoost)


(2) Regression
+ [Transfer Stacking](https://github.com/Bin-Cao/TrAdaboost/tree/main/Transfer%20Stacking)
+ [TrAdaboost R2](https://github.com/Bin-Cao/TrAdaboost/tree/main/TrAdaBoost_R2)
+ [Two stage TrAdaboost R2](https://github.com/Bin-Cao/TrAdaboost/tree/main/Two_stage_TrAdaboost_R2)
+ [Two stage TrAdaboost R2 rv](https://github.com/Bin-Cao/TrAdaboost/tree/main/Two_stage_TrAdaboost_R2_revised)

Written using Python, which is suitable for operating systems, e.g., Windows/Linux/MAC OS etc.

## Tutorial


+ [TrAdaBoost tutorial 1](./tutorial/tutorial_5_TrAdaBoost.pdf) and [TrAdaBoost tutorial 2](./tutorial/tutorial_6_TrAdaBoost_R2.pdf) by [Mr. Chen](https://github.com/georgedashen) for **AMAT 6000A: Advanced Materials Informatics (Spring 2025, HKUST-GZ).**  Thanks for his important contributions.




## 中文介绍(陆续更新）
+ [TrAdaBoost](https://mp.weixin.qq.com/s/NhxSGOHIr3s6WwffJOrIlQ)

## Note
``` javascript
author_email='bcao@shu.edu.com'
maintainer='CaoBin'
maintainer_email='bcao@shu.edu.cn' 
license='MIT License'
url='https://github.com/Bin-Cao/TrAdaboost'
python_requires='>=3.7'
```

References

.. [1] Dai, W., Yang, Q., et al. (2007). 
Boosting for Transfer Learning.(2007), 193--200. 
In Proceedings of the 24th international conference on Machine learning.

.. [2] Yao, Y., & Doretto, G. (2010, June)
Boosting for transfer learning with multiple sources. IEEE.
DOI: 10.1109/CVPR.2010.5539857

.. [3] Rettinger, A., Zinkevich, M., & Bowling, M. (2006, July). 
Boosting expert ensembles for rapid concept recall. 
In Proceedings of the National Conference on Artificial Intelligence 
(Vol. 21, No. 1, p. 464). 
Menlo Park, CA; Cambridge, MA; London; AAAI Press; MIT Press; 1999.
    
.. [4] Pardoe, D., & Stone, P. (2010, June). 
Boosting for regression transfer. 
In Proceedings of the 27th International Conference 
on International Conference on Machine Learning (pp. 863-870).


## About
Maintained by Bin Cao. Please feel free to open issues in the Github or contact Bin Cao
(bcao686@connect.hkust-gz.edu.cn) in case of any problems/comments/suggestions in using the code. 

---
## Transfer learning links
1 : Instance-based transfer learning
  - Instance selection (marginal distributions are same while conditional distributions are different) :
    
    [TrAdaboost](https://github.com/Bin-Cao/TrAdaboost/tree/main/TrAdaBoost)

  - Instance re-weighting (conditional distributions are same while marginal distributions are different) :
    
    [KMM](https://github.com/Bin-Cao/KMMTransferRegressor)
 
2 : Feature-based transfer learning
  - Explicit distance:
      - case 1 : marginal distributions are same while conditional distributions are different:
        
         [TCA(MMD based)](https://github.com/MaterialsInformaticsDemo/TCA)  ; [DAN(MK-MMD based)](https://github.com/MaterialsInformaticsDemo/DAN)
      - case 1 : conditional distributions are same while marginal distributions are different
      
        JDA
      
      - case 3 : Both marginal distributions and conditional distributions are different
      
        DDA
      
 - Implicit distance :
 
   DANN

 3 : Parameter-based transfer learning
 
 - Pretraining + fine tune

---
