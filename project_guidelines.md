# 📋 Project Guidelines — Basics of AI and Machine Learning: Final Group Project

> **Dataset in use:** Credit Card Fraud Detection (`dhruvb2028/credit-card-fraud-dataset` from Kaggle)  
> **Task:** Binary Classification (Fraud vs. Non-Fraud)  
> **Submission deadline:** 14 April 2026  
> **Presentation:** 16 April 2026 (in person)  
> **Total points:** 50 + up to 10 bonus

---

## 📁 Project File Structure

```
final_work/
├── data_collection.ipynb           # Data loading & saving as DataFrame
├── ML_work_and_analysis.ipynb      # Main project notebook (EDA → Models → Results)
├── Final_Project_Description_Basics_of_ML.docx.pdf
├── project_guidelines.md           # This file
└── data/                           # (optional) local copy of dataset CSV
```

---

## 🗺️ Step-by-Step Implementation Plan

---

### ✅ STEP 1 — Data Loading & Initial Inspection (`data_collection.ipynb`)

**Status:** Done ✔️ — all items covered in `ML_work_and_analysis.ipynb` Section 1 (`data_collection.ipynb` not created as a separate file — not required)

- [x] Download dataset via `kagglehub`
- [x] Load `credit_card_frauds.csv` into `df` (pandas DataFrame)
- [x] Print shape and preview with `df.head()`
- [x] Print `df.info()` — column names, dtypes, non-null counts
- [x] Print `df.describe()` — summary statistics
- [x] Check for missing values: `df.isnull().sum()`
- [x] Check target column name and value counts: `df['is_fraud'].value_counts()` — 0.52% fraud confirmed

---

### 📊 STEP 2 — Exploratory Data Analysis (EDA)

**Notebook:** `ML_work_and_analysis.ipynb`

#### 2.1 Basic Summary
- [x] `df.shape`, `df.dtypes`, `df.describe()`, `df.info()`
- [x] Missing value heatmap or bar chart
- [x] Target class distribution bar chart — 0.52% fraud (1,761 of 339,607)

#### 2.2 Required Visualisations (minimum per rubric)
- [x] **Histogram / KDE plots** for each numerical feature, coloured by target class
- [x] **Correlation heatmap** — Section 2.7
- [x] **Bar charts** for categorical feature distributions — Section 2.4 (merchant category), Section 2.2 (hour/day)
- [x] **Box plots or Violin plots** for at least 3–4 features — Section 2.6
- [x] *(bonus)* **Temporal fraud patterns** — fraud rate by hour and day of week (Section 2.2, added by Jolanta)
- [x] *(bonus)* **Precision-Recall curves** — all 5 models (Section 6, added by Jolanta)

#### 2.3 Class Imbalance Strategy
- Credit card fraud is highly imbalanced. Chosen:
  - [x] **SMOTE** applied to training set only — documented and justified in Section 4.3 markdown
  - [ ] ~~Undersampling majority class~~
  - [ ] ~~Class weights~~

---

### ✅ STEP 3 — Data Preprocessing & Feature Engineering

- [x] Identify categorical columns — One-Hot Encoding applied to `category` (14 values) and `state` (13 values) via `pd.get_dummies`
- [x] Identify numerical columns — **StandardScaler** applied to numeric features for Logistic Regression; tree-based models use unscaled data. Noted in Section 3 markdown.
- [x] Handle missing values — dataset has no missing values (confirmed in Section 1)
- [x] Feature engineering — 6 new features created: `hour`, `day_of_week`, `month`, `age`, `distance_km` (Haversine), `log_amt` — documented in Section 3.2
- [x] Drop irrelevant/ID columns — `merchant`, `city`, `trans_num`, `job`, `trans_date_trans_time`, `dob` dropped after extraction

---

### ✂️ STEP 4 — Train–Test Split

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

- [x] **80% training / 20% test** split
- [x] Use `stratify=y` to maintain class distribution in both splits
- [x] Use `random_state=42` for reproducibility
- [x] ⚠️ Test set untouched during tuning — SMOTE and scaler fitted on training data only

---

### 🤖 STEP 5 — Model Training & Hyperparameter Tuning

Train **at least 4 models** (5th recommended for bonus points):

| # | Model | Class | Key Hyperparameters |
|---|-------|-------|---------------------|
| 1 | Logistic Regression | `LogisticRegression` | `C`, `penalty` (l1/l2), `solver` |
| 2 | Decision Tree | `DecisionTreeClassifier` | `max_depth`, `min_samples_split`, `criterion` |
| 3 | Random Forest | `RandomForestClassifier` | `n_estimators`, `max_depth`, `max_features` |
| 4 | Gradient Boosting / XGBoost | `XGBClassifier` | `n_estimators`, `learning_rate`, `max_depth`, `subsample` |
| 5 ⭐ | CatBoost or LightGBM or KNN | `CatBoostClassifier` / `LGBMClassifier` / `KNeighborsClassifier` | varies |

#### Tuning Strategy
- Use **`GridSearchCV`** for small search spaces (e.g., Logistic Regression, Decision Tree)
- Use **`RandomizedSearchCV`** for large search spaces (Random Forest, XGBoost)
- Always use `cv=5` (5-fold cross-validation) on training data only

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Example: Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'max_features': ['sqrt', 'log2']
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train)
print(grid.best_params_)
```

- [x] Best hyperparameters saved — printed per model and summarised in Section 5 markdown cells

---

### 📈 STEP 6 — Evaluation & Comparison

After tuning, retrain each best model on full `X_train`, then evaluate on `X_test`.

#### Metrics to Compute for Each Model
- [x] **Accuracy**
- [x] **Precision, Recall, F1-Score** — `classification_report` for all 5 models
- [x] **ROC-AUC** — computed for all 5 models
- [x] **Confusion Matrix** — heatmaps for best and worst model

#### Required Visualisations
1. [x] **Grouped bar chart** — Accuracy, F1, ROC-AUC for all 5 models
2. [x] **ROC curves** — all 5 models on same axes with AUC in legend
3. [x] **Summary comparison table** — styled DataFrame with colour gradient
4. [x] **Confusion matrix heatmaps** — best and worst model
5. [x] *(bonus)* **Precision-Recall curves** — all models with random baseline
6. [x] *(bonus)* **Business impact table** — fraud caught/missed, estimated dollar losses per model
7. [x] *(bonus)* **Threshold optimisation chart** — Precision/Recall/F1 vs threshold (0.1–0.7)

```python
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
plt.title("Best Model — Confusion Matrix")
plt.show()
```

---

### ✅ STEP 7 — Model Interpretation & Analysis

- [x] **Feature importance plot** — top 25 features from best model + Random Forest comparison (Section 7)
- [x] Discuss **why** the best model outperforms the others — Sections 8.2 and 8.3 (gradient boosting captures non-linear feature interactions)
- [x] Connect findings to the domain — Section 7.1 table maps each key feature to fraud logic and EDA evidence
- [ ] ⭐ **(Optional bonus)** SHAP values — **NOT implemented**; mentioned as future work in Section 8.4

---

### ⚠️ STEP 8 — Notebook Formatting & Documentation

- [x] Every section has a **markdown header** — all 8 sections fully written with beginner-friendly explanations (rewritten by Jolanta April 11)
- [ ] ❌ **Contribution Table — MISSING** — must be added before submission (direct point loss)
- [x] **References section** — Section 8.5 (dataset, SMOTE paper, sklearn/XGBoost/LightGBM docs)
- [x] Notebook **runs end-to-end without errors** — confirmed (TODO April 12)

---

### ✅ STEP 9 — Presentation (due 14.04, presented 16.04)

**Status:** `PRESENTATION_README.md` created — 18-slide plan with speaker notes, beginner glossary for teammates, demo instructions, and pre-presentation checklist.

Recommended slide structure (10–15 min):

| Slide | Content |
|-------|---------|
| 1 | Title, team names, date |
| 2 | Problem statement — what is being predicted and why it matters |
| 3 | Dataset overview — rows, features, class balance |
| 4 | EDA highlights — 2–3 best visualisations |
| 5 | Preprocessing decisions |
| 6 | Methodology — split strategy, CV, models trained |
| 7 | Results — grouped bar chart + ROC curves |
| 8 | Best model — confusion matrix, hyperparameters, feature importance |
| 9 | Conclusions & lessons learned |
| 10 | Team contributions |
| 11 | References |

---

## 🏆 Grading Rubric

| Category | Points |
|----------|--------|
| EDA & Preprocessing | 15 |
| Model Training, Hyperparameter Tuning, Evaluation | 15 |
| Model Analysis, Interpretation, Presentation | 20 |
| **Total** | **50** |
| Bonus (extra models, SHAP, etc.) | up to +10 |

---

## ⚠️ Critical Rules

1. **No test set leakage** — never use `X_test` / `y_test` during model selection or tuning
2. **Always use `random_state=42`** for reproducibility
3. **All code must be original** — AI tools may assist but you must understand every line
4. **Cite all external sources** in a References section
5. **Notebook must run end-to-end** without errors before submission

---

## 📦 Python Libraries Needed

```python
# Already installed in environment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier  # bonus
from lightgbm import LGBMClassifier      # bonus
from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay, roc_auc_score, roc_curve)
# Optional
import shap
from imblearn.over_sampling import SMOTE
```

---

## 🔗 Useful References

- [Kaggle Dataset](https://www.kaggle.com/datasets/dhruvb2028/credit-card-fraud-dataset)
- [sklearn Documentation](https://scikit-learn.org/stable/)
- [XGBoost Docs](https://xgboost.readthedocs.io/)
- [SHAP Library](https://shap.readthedocs.io/)
- [IBM Classification Comparison Example](https://github.com/patrick013/Classification-Algorithms-with-Python)
- [CRISP-DM Classification Example](https://github.com/hemangsharma/Assignment-2---Classification-Models)
- [Tabular Data Course](https://github.com/davidrpugh/machine-learning-for-tabular-data)
- [XGBoost vs Deep Learning Paper](https://arxiv.org/abs/2106.03253)
