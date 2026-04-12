# Jolanta Edits — Change Log

---

## April 12, 2026

### Files Added
- **`PRESENTATION_README.md`** — full presentation guide: 18-slide plan, beginner-friendly project explanation for teammates (plain-English glossary of 15 terms, step-by-step project walkthrough, FAQ for likely audience questions), live demo instructions for the coursemate running Docker, and a pre-presentation checklist
  - Manually edited after creation: section header renamed from `For Teammates — What We Did (Plain English)` → `What We Did`
- **`Presentation/presentation_speech.md`** — full speaker script for the 18-slide deck with `[SLIDE X]` advance markers; includes actual model metric numbers filled in (e.g. LightGBM Recall 0.8399, XGBoost Recall 0.8764), business impact figures ($518.07 avg fraud value, ~$32–39k prevented), threshold analysis results, and a Q&A quick-reference section for common audience questions
- **`Presentation/index.html`** — standalone HTML presentation page with Chart.js charts, dark-themed UI with fixed navigation, and all project content rendered as a web page (can be opened in a browser directly, no server needed)

### Project Validation
- Ran a full validation against `project_guidelines.md` and `rules.md`
- Confirmed covered: EDA, all required visualisations, SMOTE, feature engineering, 5 models with GridSearch/RandomizedSearch, all metrics, ROC/PR curves, confusion matrices, threshold optimisation, feature importance, references
- Identified gaps: **Contribution Table** missing from notebook (required by guidelines, direct point loss), SHAP values not implemented (optional bonus)

### Testing Notebooks Added
- **`Phase_2_testing.ipynb`** — 9 tests (Tests 6–14) validating feature engineering math: haversine distance, log_amt, age calculation, hour/day/month extraction, one-hot encoding alignment, and StandardScaler application
- **`Phase_3_testing.ipynb`** — 5 tests (Tests 15–19) validating model inference: valid class label output, predict_proba in [0,1], determinism across 5 repeated calls, directional sanity (fraud pattern scores higher than legit), and batch prediction of 5 mixed rows

### `project_guidelines.md` — Full Status Review & Update
- Reviewed every checklist item against actual notebook content
- Updated all completed items to `[x]` across Steps 1–9 (previously all unchecked)
- Added bonus items achieved beyond the rubric: temporal EDA, Precision-Recall curves, business impact table, threshold optimisation chart
- Flagged **Contribution Table** as `❌ MISSING` (Step 8) — only remaining required item before submission
- Flagged SHAP as not implemented (optional bonus, mentioned as future work in Section 8.4)
- Step 9 updated to reference `PRESENTATION_README.md`

### Housekeeping
- Deleted `__pycache__/` directory from project root

---

## April 11, 2026

### Testing Notebooks Added
- **`Phase_1_testing.ipynb`** — 5 tests (Tests 1–5) validating artefact loading: model_artefacts/ directory exists, all 4 files present (lgbm_fraud_model.pkl, scaler.pkl, feature_columns.pkl, num_features.pkl), model is a LightGBM classifier, feature_cols is a list of 50+ strings, num_features is a non-empty list

### Other Changes
Changes made to `ML_work_and_analysis.ipynb` and supporting files.

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

### Section 5 — Per-model interpretation cells added
Between every model training code cell, two cells were inserted:
1. **Interpretation markdown** — explains what the printed metrics mean for *that specific model*, what its strengths and weaknesses are for fraud detection, and what to look for in the output
2. **Bridge markdown** — transitions to the next model with a brief motivation for why the next model was chosen and how it differs from the previous one

A **5.5 bridge** cell was added after LightGBM transitioning into Section 6.

---

### Section 6 — Enhanced intro (6.1 / 6.2 / 6.3)
The original Section 6 header was expanded into three sub-sections:
- **6.1** — Why evaluation matters: explains that a model that predicts "always legitimate" achieves 99.48% accuracy, so accuracy alone is meaningless
- **6.2** — Metrics reference table: Precision, Recall, F1-Score, ROC-AUC each explained with a formula, plain-English meaning, and fraud scenario example
- **6.3** — Reading the results: instructions on what to look for when comparing models in the table that follows

### Section 6 — Winner announcement cell added (after comparison table)
A new Python cell was inserted after the metrics comparison table. It:
- Rebuilds `results_df` if the preceding cell was skipped (safety guard)
- Prints the best model name, Recall score, fraud cases caught/missed out of total
- Prints the gain in Recall over the Logistic Regression baseline
- Prints the worst model and how many fraud cases it missed in comparison

### Section 6 — Bar chart takeaway added
A **📌 Bar Chart Takeaway** cell was inserted after the grouped bar chart. Explains: focus on the red Recall bars, why Accuracy barely moves across models, and what the gap between best and worst Recall means in practice.

### Section 6 — ROC curve takeaway added (after ROC curve)
A **📌 ROC Curve Takeaway** cell was inserted directly after the ROC curve plot. Explains the top-left corner target, what AUC measures, and why ROC-AUC is misleading on imbalanced data — bridging to the Precision–Recall curve below.

### Section 6 — Precision–Recall curve added + takeaway
A new Python cell plotting **Precision–Recall curves** for all models was inserted after the ROC curve. Uses `precision_recall_curve` and `average_precision_score`, includes a random baseline at the fraud prevalence rate (~0.52%). A **📌 Precision–Recall Takeaway** was added directly after, explaining why this curve is more honest on imbalanced data and how to read the Average Precision score.

### Section 6 — Confusion matrix takeaway added (after confusion matrices)
A **📌 Confusion Matrix Takeaway** cell was inserted directly after the confusion matrix heatmaps. Explains what each quadrant means, why False Negatives (missed fraud) are the most costly error type, and how to compare the two matrices.

### Section 6 — Business impact table added + takeaway
A new Python cell was inserted after the confusion matrices. It:
- Calculates average fraud transaction value from the dataset
- For each model: prints fraud caught, missed, estimated dollar loss, and false alarms
- Prints the delta between best and worst model in cases caught and estimated dollars prevented

A **📌 Business Impact Takeaway** was added after this cell, connecting Recall percentages to real financial stakes.

### Section 6.4 — Threshold Optimisation added
Two new cells were added:
1. **Markdown (6.4 intro)** — explains what a classification threshold is using a smoke alarm analogy; explains the trade-off between Recall (catching fraud) and Precision (avoiding false alarms) at different threshold settings
2. **Python cell** — tests the best model at thresholds 0.1–0.7, prints a table of Precision/Recall/F1/Fraud caught/False alarms, plots a line chart of all three metrics vs threshold, and prints the threshold that maximises F1

---

*April 11, 2026*
