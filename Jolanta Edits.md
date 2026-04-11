# Jolanta Edits — Change Log

Changes made to `ML_work_and_analysis.ipynb` and supporting files — **April 11, 2026**.

---

## Files Added

- **`TODO.md`** — prioritised list of 5 incomplete/missing items in the project (empty `model_test.py`, notebook re-run, threshold optimisation, API security gaps, missing `data_collection.ipynb`)
- **`Jolanta Edits.md`** — this file

---

## Notebook Changes

### Section 1
- Added a new cell printing a concise dataset summary (total rows, columns, numeric vs categorical count)

### Section 2.1 — EDA Introduction (markdown rewritten)
- Expanded from a minimal paragraph to a full beginner-friendly introduction
- Defines EDA, explains the target variable `is_fraud`, quantifies the imbalance (339,607 rows, only 1,761 fraud at 0.52%), explains why accuracy is a misleading metric here, and defines Recall, F1-Score, and ROC-AUC in plain English
- Mentions SMOTE as the imbalance fix applied later in Section 4

### Section 2 — New subsection 2.2 added (Temporal Fraud Patterns)
- Two new cells: a markdown explanation + a Python cell producing bar charts of fraud rate by **hour of day** and **day of week**, with average lines and a printed top-5 peak hours list

### Section 2 — Subsections reordered by complexity

| New | Title | Was |
|-----|-------|-----|
| 2.1 | Target distribution | 2.1 |
| 2.2 | Time of day & day of week | *(new)* |
| 2.3 | Transaction amount | 2.2 |
| 2.4 | Merchant category | 2.3 |
| 2.5 | Geographic distribution | 2.4 |
| 2.6 | Violin & box plots | 2.6 |
| 2.7 | Correlation heatmap | 2.5 |

Heatmap moved to last — it is the most abstract view and makes more sense once each feature has been seen individually.

### Section 2 — Takeaway cells added
A **📌 Takeaway** cell was added after each of the 7 EDA subsections summarising the key observation, why it matters, and what happens next in the notebook.

### Section 3 — Full markdown rewrite (beginner-friendly)
The original Section 3 header was a brief strategy table. It was fully rewritten with three sub-sections:
- **3.1** — Column-by-column table showing what happens to each original column and *why* (with references back to EDA findings)
- **3.2** — Each newly engineered feature (`hour`, `day_of_week`, `month`, `age`, `distance_km`, `log_amt`) explained: how it is built and what fraud signal it captures
- **3.3** — One-Hot Encoding explained with a concrete example showing how `category` becomes 14 binary columns

### Section 4 — Full markdown rewrite (beginner-friendly)
The original Section 4 header was a short bullet list. Fully rewritten with three sub-sections:
- **4.1** — Train–Test Split explained using the exam study analogy; explains `stratify=y` and `random_state=42`
- **4.2** — Feature Scaling (StandardScaler) explained with concrete examples of mismatched column scales; includes a data leakage warning about fitting the scaler only on training data
- **4.3** — SMOTE explained with a before/after table showing the class balance change, how synthetic fraud is created using nearest neighbours, and why the test set must never be oversampled

### Section 5 — Full markdown rewrite + new sub-section (beginner-friendly)
The original Section 5 header was a model table. Fully rewritten with four sub-sections:
- **5.1** — All 5 models explained in plain English with complexity ratings and analogies (e.g. "hiring 5 detectives", "the crowd is wiser than any single tree")
- **5.2** — Hyperparameter tuning explained: what hyperparameters are, why wrong settings cause overfitting or underfitting, how RandomizedSearchCV works step by step, and why F1-score is used for tuning
- **5.3** — Implementation details: why Logistic Regression uses GridSearchCV vs Randomized, why XGBoost skips SMOTE in favour of `scale_pos_weight`, what `cv=5` and `n_jobs=-1` mean
- **5.4 (new)** — "Model Training" header added before the first model code cell. Includes a detailed description of **Logistic Regression**: what it is, how the sigmoid formula works, why it is used as a baseline, its limitations for fraud detection (cannot capture non-linear feature combinations), and each hyperparameter tuned (`C`, `solver`, `class_weight`) explained

---

*April 11, 2026*
