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

---

*April 11, 2026*
