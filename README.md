# DataScience_MachineLearning----Randomization_and_SVM----BankLoan_Prediction_project
# Telecom Customer Segmentation & Predictive Profiles (K-Means, PCA & Grid Search)

This repository hosts an advanced machine learning case study designed to segment a telecommunications customer base and build predictive classification layers for new subscribers. Moving beyond standard classification, this pipeline leverages **K-Means Clustering** to establish optimal behavioral segments, applies **Principal Component Analysis (PCA)** to resolve multi-collinearity, and implements hyperparameter-tuned estimators via **GridSearchCV** to map new subscribers (`Telco_new_cust.csv`) to target customer profiles.

---

## 📂 Dataset Repository & Ecosystem

The analytical framework combines core customer billing records, unseen target profiles, and reduced feature component projections:

1. **`telco_csv.csv` (Historical Baseline):** Contains baseline demographic and behavioral records used to discover and train cluster parameters. Features include `tenure`, `age`, `income`, `marital`, and detailed utility usage indicators (`tollfree`, `equip`, `callcard`, `wireless`, `longmon`, `tollmon`, etc.).
2. **`Telco_new_cust.csv` (Inference Vector):** Unlabeled data representing new or incoming subscribers to be profiled and assigned to established service brackets.
3. **`new_tel_com_pca_model.csv` (Dimension Output):** The final engineered dataset consisting of the top **10 Principal Components (`PC1` to `PC10`)** mapped alongside the final cluster `prediction` identifiers.

---

## 🛠️ Unsupervised Unification & Feature Engineering

The workflow in `TeleCom_CustomerPrediction.ipynb` follows a strict unsupervised feature extraction process:

### 1. Feature Standardization
Because network usage metrics (e.g., long-distance minutes, monthly equipment bills) use vastly different scales compared to demographic variables (e.g., age, household size), scikit-learn's `StandardScaler` is applied to equalize feature variances before computing distances.

### 2. K-Means Clustering Optimization
* The mathematical customer groups (`custcat`) are segmented using distance vectors. 
* **Validation Bounds:** Segment counts are validated using **Silhouette Scores** (`silhouette_score`) to maximize tight cluster density while keeping cluster centers clearly separated.

### 3. Dimensionality Reduction (PCA)
To strip out high multi-collinearity among highly correlated billing indicators (such as `tollmon`, `equipmon`, `cardmon`, and `wiremon`), orthogonal transformation is used to compress the feature space into **10 Principal Components** that preserve the highest possible data variance.

---

## 🤖 Predictive Modeling & Hyperparameter Optimization

Once the target behavioral clusters are established, the framework switches to a supervised classification pipeline to assign new profiles to their correct categories:

* **Support Vector Machine Classifier (SVC):** Applied over the reduced orthogonal components to find clear margins between customer segments.
* **Tuning via
