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

**Status:** Done ✔️

- [x] Download dataset via `kagglehub`
- [x] Load `credit_card_frauds.csv` into `df` (pandas DataFrame)
- [x] Print shape and preview with `df.head()`

**What still needs to be done here:**
- [ ] Print `df.info()` — column names, dtypes, non-null counts
- [ ] Print `df.describe()` — summary statistics
- [ ] Check for missing values: `df.isnull().sum()`
- [ ] Check target column name and value counts: `df['target_col'].value_counts()`

---

### 📊 STEP 2 — Exploratory Data Analysis (EDA)

**Notebook:** `ML_work_and_analysis.ipynb`

#### 2.1 Basic Summary
- [ ] `df.shape`, `df.dtypes`, `df.describe()`, `df.info()`
- [ ] Missing value heatmap or bar chart
- [ ] Target class distribution bar chart — check for class imbalance (fraud datasets are usually ~0.1–2% fraud)

#### 2.2 Required Visualisations (minimum per rubric)
- [ ] **Histogram / KDE plots** for each numerical feature, coloured by target class
- [ ] **Correlation heatmap** — `sns.heatmap(df.corr(), annot=True)`
- [ ] **Bar charts** for categorical feature distributions
- [ ] **Box plots or Violin plots** for at least 3–4 features (to show spread & outliers)

#### 2.3 Class Imbalance Strategy
- Credit card fraud is highly imbalanced. Choose one:
  - [ ] **SMOTE** (oversampling minority class) — `from imblearn.over_sampling import SMOTE`
  - [ ] **Undersampling** majority class
  - [ ] **Class weights** — `class_weight='balanced'` in sklearn models
- Document and justify the choice in a markdown cell.

---

### 🔧 STEP 3 — Data Preprocessing & Feature Engineering

- [ ] Identify categorical columns — apply **One-Hot Encoding** (`pd.get_dummies` or `OneHotEncoder`)
- [ ] Identify numerical columns — apply **StandardScaler** or **MinMaxScaler** where needed
  - Note in notebook: tree-based models (Decision Tree, Random Forest, XGBoost) **do NOT need scaling**; Logistic Regression **does**
- [ ] Handle missing values:
  - Numerical: impute with **mean** or **median**
  - Categorical: impute with **mode** or add an "Unknown" category
- [ ] (Optional) Feature engineering — e.g., transaction amount ratios, time-based binning
- [ ] Drop irrelevant/ID columns (e.g., row IDs, timestamps if not useful)

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

- [ ] **80% training / 20% test** split
- [ ] Use `stratify=y` to maintain class distribution in both splits
- [ ] Use `random_state=42` for reproducibility
- [ ] ⚠️ **NEVER touch the test set until final evaluation** — all tuning happens on training data only

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

- [ ] Save best hyperparameters for each model in a **summary table** (markdown or DataFrame)

---

### 📈 STEP 6 — Evaluation & Comparison

After tuning, retrain each best model on full `X_train`, then evaluate on `X_test`.

#### Metrics to Compute for Each Model
- [ ] **Accuracy**
- [ ] **Precision, Recall, F1-Score** — `classification_report(y_test, y_pred)`
- [ ] **ROC-AUC** — `roc_auc_score(y_test, y_proba)`
- [ ] **Confusion Matrix** — visualised as heatmap

#### Required Visualisations
1. [ ] **Grouped bar chart** — Accuracy, F1, ROC-AUC for all models side-by-side
2. [ ] **ROC curves** — all models on the same axes, with AUC in the legend
3. [ ] **Summary comparison table** — all metrics in one DataFrame
4. [ ] **Confusion matrix heatmaps** — at minimum for best and worst model

```python
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
plt.title("Best Model — Confusion Matrix")
plt.show()
```

---

### 🔍 STEP 7 — Model Interpretation & Analysis

- [ ] **Feature importance plot** from the best tree-based model (Random Forest or XGBoost)
  ```python
  importances = best_model.feature_importances_
  # Plot as horizontal bar chart
  ```
- [ ] Discuss **why** the best model outperforms the others (complexity, regularisation, ensemble effect)
- [ ] Connect findings to the domain: which features are most predictive of fraud?
- [ ] ⭐ **(Optional bonus)** SHAP values:
  ```python
  import shap
  explainer = shap.TreeExplainer(best_model)
  shap_values = explainer.shap_values(X_test)
  shap.summary_plot(shap_values, X_test)
  ```

---

### 📝 STEP 8 — Notebook Formatting & Documentation

- [ ] Every section must have a **markdown header** explaining what is being done and why
- [ ] Include a **Contribution Table** at the top or end:

| Team Member | Sections Responsible |
|-------------|----------------------|
| Name 1 | EDA, Visualisations |
| Name 2 | Preprocessing, Feature Engineering |
| Name 3 | Model Training, Tuning |
| Name 4 | Evaluation, Interpretation |
| Name 5 | Presentation, Report |

- [ ] Include a **References section** at the end with dataset source and any code references
- [ ] Ensure notebook **runs end-to-end without errors** before submission

---

### 📊 STEP 9 — Presentation (due 14.04, presented 16.04)

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
