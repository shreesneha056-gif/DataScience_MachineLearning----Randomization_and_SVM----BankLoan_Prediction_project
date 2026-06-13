# Bank Loan Default Prediction: Class Imbalance Resolution & Support Vector Machine (SVM) Classification

This repository hosts a data science case study focused on predicting bank loan defaults (`default`). Because default events are naturally rare compared to standard bank operations, the core objective of this project is to address **severe class imbalance** using advanced resampling techniques before training a **Support Vector Classifier (SVC)** to maximize detection metrics ($F_1\text{-score}$, Recall) for high-risk customers.

---

## 📂 Dataset Repository & Features

The model trains on historical credit risk assessment records mapping financial behavior profiles against loan outcomes.

### Dataset Profile (`bankloans.csv`)
* **Target Variable:** `default` (Binary indicator: `1` if the customer defaulted, `0` if they paid back successfully).
* **Feature Attributes:**
  - `age`: Age of the customer.
  - `ed`: Education level.
  - `employ`: Years with current employer.
  - `address`: Years at current residential address.
  - `income`: Annual customer income.
  - `debtinc`: Debt-to-income ratio (%).
  - `creddebt`: Credit card debt.
  - `othdebt`: Other outstanding debts.

---

## ⚖️ The Class Imbalance Challenge & Resampling Framework

Standard classifiers like Support Vector Machines try to maximize overall accuracy, leading them to ignore the minority class (default cases). The workspace in `imbalance_SVC.ipynb` evaluates and benchmarks three distinct strategies from the `imblearn` library to fix this bias:

1. **Random Under-Sampling (`RandomUnderSampler`):** Down-samples the majority class (`0`) by randomly removing records until it matches the minority class count. 
2. **Random Over-Sampling (`RandomOverSampler`):** Replicates minority class instances (`1`) at random to increase their weight in the decision bounds.
3. **SMOTE (Synthetic Minority Over-sampling Technique):** Mathematically generates *synthetic* data points along the line segments joining k-nearest neighbors of the minority class, expanding the minority class feature space without creating direct duplicates.

---

## 🤖 Support Vector Classifier (SVC) Modeling Pipeline

The classification engine relies on a baseline and tuned **Support Vector Machine (SVM)** to draw clean margins between safe and defaulting applicants:

### 1. Vector Margin Mechanics
* The algorithm maps features into high-dimensional space to identify an optimal hyper-plane that maximizes the physical separation margin between target boundaries.

### 2. Performance Tracking & Baseline Discrepancy
An initial evaluation highlight is documented directly within the notebook's classification matrix reports:
* **The Problem:** Imbalanced baseline training leads to a massive disparity—yielding a high precision/recall for class `0.0` (~0.99 recall) but a weak $F_1\text{-score}$ of **0.22** and a recall of **0.13** for class `1.0`.
* **The Solution:** Applying resampled data distributions transforms the boundary matrix, raising minority class recall metrics significantly to protect the bank from hidden default risks.

```python
# Extract from pipeline: Evaluating performance boundaries via classification report
print(classification_report(y_test, svc.predict(X_test)))