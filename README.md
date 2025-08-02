

# TrAdaBoost: Boosting for Transfer Learning

🤝🤝🤝 **Please star ⭐️ this project to support open-source research and development 🌍! Thank you!**

This is a teaching and research-oriented project that implements **transfer learning using boosting strategies**, developed during my stay at [Zhejiang Lab](https://www.zhejianglab.org/lab/home) (March 1 – August 31, 2023).
If you have any questions or need assistance, feel free to reach out!

---

## 🔬 Overview

**Transfer learning** aims to leverage knowledge from one or more **source domains** to improve performance on a **target domain** with limited data. This project focuses on instance-based methods, particularly variants of the **TrAdaBoost** algorithm for both classification and regression tasks.

[![Security Status](https://www.murphysec.com/platform3/v3/badge/1626904646967132160.svg)](https://www.murphysec.com/accept?code=645babf2266d3ebb42b1005074b53306&type=1&from=2)

---

## 📦 Models Included

### 🔹 Classification

* [TrAdaBoost](https://github.com/Bin-Cao/TrAdaboost/blob/main/TrAdaBoost)
* [MultiSourceTrAdaBoost](https://github.com/Bin-Cao/TrAdaboost/blob/main/MultiSourceTrAdaBoost)
* [TaskTrAdaBoost](https://github.com/Bin-Cao/TrAdaboost/blob/main/TaskTrAdaBoost)
* [ExpBoost](https://github.com/Bin-Cao/TrAdaboost/tree/main/ExpBoost)
* [Improved ExpBoost](https://github.com/Bin-Cao/TrAdaboost/tree/main/Improved%20ExpBoost)

### 🔸 Regression

* [Transfer Stacking](https://github.com/Bin-Cao/TrAdaboost/tree/main/Transfer%20Stacking)
* [TrAdaBoost.R2](https://github.com/Bin-Cao/TrAdaboost/tree/main/TrAdaBoost_R2)
* [Two-stage TrAdaBoost.R2](https://github.com/Bin-Cao/TrAdaboost/tree/main/Two_stage_TrAdaboost_R2)
* [Revised Two-stage TrAdaBoost.R2](https://github.com/Bin-Cao/TrAdaboost/tree/main/Two_stage_TrAdaboost_R2_revised)

> Implemented in **Python**, supporting Windows, Linux, and macOS platforms.

---

## 📚 Tutorial

* 📘 [Tutorial 1: TrAdaBoost](./tutorial/tutorial_5_TrAdaBoost.pdf)
* 📘 [Tutorial 2: TrAdaBoost.R2](./tutorial/tutorial_6_TrAdaBoost_R2.pdf)
  *By [Mr. Chen](https://github.com/georgedashen), for **AMAT 6000A: Advanced Materials Informatics (Spring 2025, HKUST-GZ)**.*
  Thanks to Mr. Chen for his valuable contributions!

---

## 📈 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Bin-Cao/TrAdaboost\&type=Date)](https://star-history.com/#Bin-Cao/TrAdaboost&Date)

---

## 📌 中文介绍（持续更新）

* 📄 [TrAdaBoost 推广与介绍](https://mp.weixin.qq.com/s/NhxSGOHIr3s6WwffJOrIlQ)

---

## 📎 Citation

If you use this code in your research, please cite:

> **Cao Bin**, **Zhang Tong-yi**, **Xiong Jie**, **Zhang Qian**, **Sun Sheng**.
> *Package of Boosting-based transfer learning* \[2023SR0525555], 2023, Software Copyright.
> GitHub: [github.com/Bin-Cao/TrAdaboost](https://github.com/Bin-Cao/TrAdaboost)

---

## 🔧 Package Info

```python
author_email='bcao@shu.edu.com'
maintainer='CaoBin'
maintainer_email='bcao@shu.edu.cn'
license='MIT License'
url='https://github.com/Bin-Cao/TrAdaboost'
python_requires='>=3.7'
```

---

## 📚 References

1. Dai, W., Yang, Q., et al. (2007). **Boosting for Transfer Learning**. ICML.
2. Yao, Y., & Doretto, G. (2010). **Boosting for Transfer Learning with Multiple Sources**. CVPR.
3. Rettinger, A., et al. (2006). **Boosting Expert Ensembles for Rapid Concept Recall**. AAAI.
4. Pardoe, D., & Stone, P. (2010). **Boosting for Regression Transfer**. ICML.

---

## 💡 Related Transfer Learning Methods

### 1️⃣ Instance-based Transfer Learning

* **Instance Selection** (same marginal, different conditional distributions):
  [TrAdaBoost](https://github.com/Bin-Cao/TrAdaboost/tree/main/TrAdaBoost)

* **Instance Re-weighting** (same conditional, different marginal distributions):
  [KMM](https://github.com/Bin-Cao/KMMTransferRegressor)

### 2️⃣ Feature-based Transfer Learning

* **Explicit Distance-based**

  * Same marginal, different conditional:
    [TCA (MMD-based)](https://github.com/MaterialsInformaticsDemo/TCA) | [DAN (MK-MMD-based)](https://github.com/MaterialsInformaticsDemo/DAN)
  * Same conditional, different marginal: JDA
  * Both distributions different: DDA

* **Implicit Distance-based**

  * DANN

### 3️⃣ Parameter-based Transfer Learning

* Pretraining + Fine-tuning

---

## 🙋 About

Maintained by **Bin Cao**.
Feel free to open GitHub issues or reach out to me at:

📫 Email: `bcao686@connect.hkust-gz.edu.cn`

