# 🏦 Credit Card Fraud Detection — Presentation Guide

> **Course:** Basics of AI and Machine Learning — Final Project  
> **Presentation date:** 16 April 2026 (in person)  
> **Dataset:** Credit Card Transactions – Western United States (Kaggle, 339,607 rows)  
> **Task:** Binary Classification — predict `is_fraud` (0 or 1)

---

## 🧑‍🎓 What We Did 

Read this section first. It explains the entire project without jargon so you can confidently present any slide.

### The Big Picture

We built a **fraud detector**. A credit card company gave us a spreadsheet with ~340,000 past transactions. Each row is one purchase — who bought what, where, when, and how much. One column (`is_fraud`) says whether it turned out to be fraud or not. Our job: teach a computer to look at a *new* transaction and guess if it's fraud.

### Why It's Hard

Only **0.52%** of all transactions are fraud (1,761 out of 339,607). That's like finding 5 bad apples in a crate of 1,000. If the computer just says "everything is fine" every time, it's technically right 99.5% of the time — but it catches **zero** fraud. So the usual "accuracy" number is useless here. Instead we measure **Recall** — out of all real fraud, what percentage did we catch?

### What We Did Step by Step

| Step | What happened | Why (in simple terms) |
|------|--------------|----------------------|
| **1. Loaded data** | Downloaded the spreadsheet from Kaggle, opened it in Python | Getting the raw material |
| **2. Explored the data (EDA)** | Made charts to understand patterns: when does fraud happen? How much money? Which shops? Where on the map? | You can't build a good detector without understanding what fraud *looks like* |
| **3. Created new columns** | From the raw data we calculated: hour of day, distance between buyer and shop (km), age of buyer, log of amount | These turned out to be the most useful clues for the model — e.g. "a $4,000 purchase at 3 AM from a shop 8,000 km away" screams fraud |
| **4. Split the data** | 80% for training (the model learns from this), 20% for testing (we check if it learned well — the model never sees these during training) | Like studying with practice exams and then taking a real exam |
| **5. Fixed the imbalance (SMOTE)** | The training set had too few fraud examples, so we generated synthetic (fake but realistic) fraud rows to balance it to 50/50 | Without this, the model barely learns what fraud looks like because it sees so few examples |
| **6. Trained 5 models** | Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM — each is a different algorithm that learns patterns differently | We tried multiple approaches to find the best one |
| **7. Tuned each model** | Tested different settings (hyperparameters) using cross-validation — automatically trying hundreds of combos and keeping the best | Like adjusting the sensitivity on a smoke alarm — too sensitive = false alarms, too low = misses fires |
| **8. Compared results** | Measured Recall, F1-Score, ROC-AUC for each model on the test set. Made charts, confusion matrices, and a business impact table | Figured out which model catches the most fraud with fewest false alarms |
| **9. Picked a winner** | LightGBM (or XGBoost) came out on top — gradient boosting models are best at this type of data | They learn by correcting their own mistakes, tree after tree |
| **10. Tested the pipeline** | 19 automated tests across 3 notebooks to verify everything works correctly | Quality check before submission |
| **11. Built a demo** | Flask API + Next.js web UI running in Docker — generates fake transactions and scores them in real time | Shows the model working live during the presentation |

### Key Terms You Might Need to Explain

| Term | What it means | Analogy |
|------|--------------|---------|
| **Recall** | Of all real fraud cases, what % did we catch? | A security guard who catches 90% of shoplifters has 90% recall |
| **Precision** | Of all transactions we *flagged* as fraud, what % actually were? | If the guard flags 100 people and 80 were actually stealing → 80% precision |
| **F1-Score** | A single number that balances Recall and Precision | An overall "grade" for the guard |
| **ROC-AUC** | How well the model ranks fraud higher than legit, overall | A score from 0.5 (random guessing) to 1.0 (perfect) |
| **False Positive** | We said fraud, but it was actually legit | Blocking someone's real grocery purchase — annoying but harmless |
| **False Negative** | We said legit, but it was actually fraud | **The dangerous one** — a thief gets away |
| **SMOTE** | Creates synthetic fraud examples so the model has enough to learn from | Like photocopying practice exam questions so you have more to study from |
| **Feature Engineering** | Creating new useful columns from existing data | Raw data says "lat: 39.7, merchant_lat: 48.8" — we calculate "distance: 8,900 km" |
| **Hyperparameter Tuning** | Testing different model settings to find the best combination | Adjusting the brightness/contrast on a TV until the picture is clearest |
| **Gradient Boosting** | A technique where each new tree fixes the mistakes of the previous one | Like a team of editors, each one correcting what the last one missed |
| **One-Hot Encoding** | Converting categories (like "grocery", "travel") into separate 0/1 columns | Models only understand numbers, not words |
| **StandardScaler** | Rescaling numbers so they're all on a similar range | Makes sure "amount in dollars" (0–5000) doesn't overpower "hour" (0–23) just because of bigger numbers |
| **Train/Test Split** | Holding back 20% of data as a "final exam" the model has never seen | Ensures the model works on new data, not just data it memorised |
| **Cross-Validation (cv=5)** | During tuning, splits training data into 5 chunks and trains 5 times, each time testing on a different chunk | Like taking 5 mini-exams instead of one, for a more reliable score |
| **Threshold** | The cutoff probability above which we call something fraud (default: 0.5) | Lowering it to 0.3 = more cautious (catches more fraud, more false alarms) |

### If Someone Asks You a Question

**"Why not just use accuracy?"**
→ Because the data is 99.5% legitimate. A model that always says "not fraud" gets 99.5% accuracy but catches zero fraud. Recall tells us what actually matters: how many fraud cases we caught.

**"Why 5 models instead of just one?"**
→ The assignment requires at least 4. More importantly, no model is universally best — comparing them shows which technique works for *this* specific problem.

**"Why does the best model win?"**
→ Gradient boosting (LightGBM/XGBoost) captures combinations of features (e.g. high amount + night-time + online + far away = fraud) that simpler models like Logistic Regression can't express with a single line.

**"What is SMOTE and why do we need it?"**
→ The model originally saw 1,761 fraud examples vs 271,000 legit. It barely learned what fraud looks like. SMOTE creates realistic synthetic fraud examples so the model gets a balanced training diet. We only apply it to training data — the test set stays real.

**"What does the demo show?"**
→ The model running live: generate a fake transaction → the API scores it → the UI shows the fraud probability and verdict. It proves the model actually works outside the notebook.

---

## Slide Plan

### Slide 1 — Title

- **Title:** Credit Card Fraud Detection with Machine Learning
- **Subtitle:** Basics of AI and Machine Learning — Final Group Project, April 2026
- Team member names

---

### Slide 2 — The Problem

- A new credit card company in the western US wants to detect fraud at transaction time
- **Business requirement:** err on the side of caution — flag suspicious transactions even at the cost of false alarms
- Key metric priority: **Recall** (catch as many real fraud cases as possible)

---

### Slide 3 — The Dataset

| Property | Value |
|----------|-------|
| Rows | 339,607 transactions |
| Columns | 15 (customer, merchant, location, amount, time, target) |
| Fraud rate | **0.52%** — only 1,761 fraud cases out of 339,607 |
| Source | Kaggle (`dhruvb2028/credit-card-fraud-dataset`) |

**Key point:** A model that predicts "always legitimate" scores 99.48% accuracy but catches **zero fraud**. Accuracy is a useless metric here.

---

### Slide 4 — EDA Highlights (pick 3–4 visuals from notebook)

Findings from Section 2 of the notebook:

1. **Class imbalance** — 99.48% legit vs 0.52% fraud (bar chart, Section 2.1)
2. **Temporal patterns** — fraud peaks between 1–4 AM; weekdays and weekends are similar (Section 2.2)
3. **Amount** — fraud transactions have a clearly higher median amount than legitimate ones (Section 2.3)
4. **Merchant category** — `grocery_pos`, `shopping_net`, `misc_net` have the highest fraud rates (Section 2.4)
5. **Distance** — fraudulent merchants are much further from the cardholder's location (Section 2.5)
6. **Correlation heatmap** — shows which features relate to each other (Section 2.7)

**Suggested visuals:** fraud rate by hour bar chart, amount distribution (violin plot), correlation heatmap

---

### Slide 5 — Feature Engineering

We created **6 new features** from raw data — these ended up being among the model's most important signals:

| Feature | How it's built | Fraud signal |
|---------|---------------|--------------|
| `hour` | Extracted from transaction datetime | Fraud peaks at night (1–4 AM) |
| `day_of_week` | Extracted from datetime | Weekday vs weekend patterns |
| `month` | Extracted from datetime | Seasonal patterns |
| `age` | Transaction date minus date of birth | Certain age groups targeted more |
| `distance_km` | Haversine formula: cardholder lat/long → merchant lat/long | Large distance = card likely stolen |
| `log_amt` | `log(1 + amount)` | Stabilises the extreme range of amounts, reduces outlier impact |

**Also done:**
- Dropped non-predictive columns (merchant name, city, trans_num, job)
- One-hot encoded `category` (14 values) and `state` (13 values)
- StandardScaler on numeric features (for Logistic Regression)

---

### Slide 6 — Data Preparation

| Step | Detail | Why |
|------|--------|-----|
| Train/Test split | 80% train / 20% test, `stratify=y`, `random_state=42` | Reproducible, preserves class ratio |
| Scaling | StandardScaler fitted on **train only**, applied to both | Avoids data leakage |
| SMOTE | Applied to **training set only** — rebalanced from 0.52% → 50% fraud | Gives models enough fraud examples to learn from; test set stays untouched |

---

### Slide 7 — Models Trained (5 models)

| # | Model | Tuning method | Key hyperparameters tuned |
|---|-------|--------------|--------------------------|
| 1 | Logistic Regression | GridSearchCV (cv=5) | C, solver, class_weight |
| 2 | Decision Tree | GridSearchCV (cv=5) | max_depth, min_samples_split, criterion |
| 3 | Random Forest | RandomizedSearchCV (cv=5) | n_estimators, max_depth, max_features |
| 4 | XGBoost | RandomizedSearchCV (cv=5) | n_estimators, learning_rate, max_depth, subsample, scale_pos_weight |
| 5 | LightGBM | RandomizedSearchCV (cv=5) | n_estimators, learning_rate, max_depth, num_leaves, subsample |

All tuning used **F1-Score** as the optimisation metric (balances precision and recall).

---

### Slide 8 — Results Comparison

**Show the grouped bar chart from Section 6** (Accuracy, F1, ROC-AUC side by side for all 5 models)

Key results to highlight (fill in exact numbers from your notebook output):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| LightGBM | ___.__ | ___.__ | ___.__ | ___.__ | ___.__ |
| XGBoost | ___.__ | ___.__ | ___.__ | ___.__ | ___.__ |
| Random Forest | ___.__ | ___.__ | ___.__ | ___.__ | ___.__ |
| Decision Tree | ___.__ | ___.__ | ___.__ | ___.__ | ___.__ |
| Logistic Regression | ___.__ | ___.__ | ___.__ | ___.__ | ___.__ |

**Key message:** Gradient boosting models (LightGBM / XGBoost) had the best Recall — they caught the most fraud.

---

### Slide 9 — ROC & Precision-Recall Curves

**Show two plots from Section 6:**
1. ROC curves — all 5 models on one chart (emphasise the top-left corner = best)
2. Precision-Recall curves — more honest on imbalanced data than ROC

**Key message:** ROC-AUC is high for all models (>0.95), but the Precision-Recall curve reveals more differences because it focuses on the rare fraud class.

---

### Slide 10 — Confusion Matrices

**Show the confusion matrix heatmaps** (best vs worst model from Section 6)

Key interpretation:
- **True Positives** (bottom-right) = correctly caught fraud
- **False Negatives** (bottom-left) = missed fraud — **the most dangerous error**
- **False Positives** (top-right) = false alarms — annoying but manageable
- The best model has fewer False Negatives than the worst

---

### Slide 11 — Business Impact

From the business impact analysis in Section 6:
- Average fraud transaction value: ~$___ (fill from notebook)
- Best model catches ___/___  fraud cases
- Worst model catches ___/___ fraud cases
- Choosing the best model over the worst prevents ~$___,___ in estimated fraud losses

**This makes model selection a business decision, not just a technical one.**

---

### Slide 12 — Threshold Optimisation

From Section 6.4:
- Default threshold is 0.5 — transaction flagged as fraud if `predict_proba ≥ 0.5`
- Lowering threshold (e.g. 0.3) → catches more fraud (higher Recall) but more false alarms
- Raising threshold (e.g. 0.7) → fewer false alarms but misses some fraud
- **Show the Precision/Recall/F1 vs Threshold chart**

Business implication: threshold is tuneable based on risk appetite.

---

### Slide 13 — Feature Importance

**Show the top-25 feature importance chart from Section 7**

Top features expected:
1. `amt` / `log_amt` — large purchases signal fraud
2. `distance_km` — merchant far from cardholder
3. `hour` — night-time transactions
4. `age` — certain demographics targeted
5. `category_*` — certain merchant types risky

**Key message:** Our engineered features (distance, hour, log_amt, age) ranked in the top — confirming that feature engineering added real predictive value.

---

### Slide 14 — Why the Best Model Wins

- **Logistic Regression** — linear boundary, cannot capture feature interactions (e.g. "high amount + night + online")
- **Decision Tree** — captures non-linearity but overfits on a single tree
- **Random Forest** — ensemble of trees, more stable, but all trees are independent
- **XGBoost / LightGBM** — gradient boosting: each tree corrects the previous tree's errors. Captures subtle multi-feature fraud patterns that simpler models miss.

---

### Slide 15 — Testing & Validation (3 Phase test suite)

| Phase | Notebook | Tests | What we validated |
|-------|----------|-------|-------------------|
| 1 | `Phase_1_testing.ipynb` | Tests 1–5 | All model artefact files load correctly (model, scaler, feature columns, numeric features) |
| 2 | `Phase_2_testing.ipynb` | Tests 6–14 | Feature engineering math is correct (haversine, log_amt, scaling, OHE alignment, age, hour/day/month) |
| 3 | `Phase_3_testing.ipynb` | Tests 15–19 | Model inference: valid outputs, deterministic, directionally sensible, batch-capable |

**All 19 tests pass ✅** — model pipeline is validated end-to-end.

---

### Slide 16 — Live Demo (coursemate runs this)

**Setup:**
```bash
docker-compose up --build
```
This starts:
- **API** on `http://localhost:5050` (Flask + LightGBM model)
- **UI** on `http://localhost:3000` (Next.js fraud detection dashboard)

**Demo flow:**
1. Open the UI at `localhost:3000`
2. Click "Generate Transaction" — creates a synthetic transaction
3. Click "Predict" — sends it to the API, shows fraud probability and verdict
4. Show a few legit transactions (low probability, ✅ Legitimate)
5. Force a fraud pattern (`?fraud=1`) — show high probability, 🚨 Fraud verdict
6. Point out the engineered features displayed (hour, distance, log_amt, age)

**API endpoints for manual demo:**
- `GET /health` — health check
- `GET /generate` — random synthetic transaction
- `GET /generate?fraud=1` — force fraudulent pattern
- `POST /predict` — score a transaction (JSON body)

---

### Slide 17 — Limitations & Future Work

**Limitations:**
- Random train/test split — no temporal validation (real fraud evolves over time)
- Haversine distance is straight-line, not road distance
- Single dataset — production needs continuous retraining

**Future work:**
- SHAP values for individual prediction explanations
- Time-series cross-validation (`TimeSeriesSplit`)
- Deep learning approaches (TabNet, FT-Transformer)
- Cost-sensitive learning (asymmetric loss function)
- Real-time streaming pipeline

---

### Slide 18 — Summary / Q&A

**Key takeaways:**
1. Accuracy is meaningless on imbalanced data — use Recall, F1, ROC-AUC
2. Feature engineering (distance, hour, log_amt, age) was as important as model choice
3. Gradient boosting (LightGBM/XGBoost) dominates tabular fraud detection
4. SMOTE + correct data leakage prevention is critical
5. Threshold tuning gives the business additional control over false alarm vs fraud catch rate

---

## Quick Reference: Where to Find Things

| What | Where |
|------|-------|
| Full analysis notebook | `ML_work_and_analysis.ipynb` |
| Phase 1 tests (artefact loading) | `Phase_1_testing.ipynb` |
| Phase 2 tests (feature engineering) | `Phase_2_testing.ipynb` |
| Phase 3 tests (model inference) | `Phase_3_testing.ipynb` |
| Flask API | `api.py` |
| UI app | `fraud-ui/` |
| Docker setup | `docker-compose.yml` |
| Model artefacts | `model_artefacts/` |
| Project guidelines | `project_guidelines.md` |
| Change log | `Jolanta Edits.md` |

---

## ⚠️ Before Presentation

- [ ] Fill in the exact metric numbers in Slide 8 and Slide 11 from the notebook output
- [ ] Run `docker-compose up --build` **before** the presentation to confirm demo works
- [ ] Pick 4–5 notebook plots to screenshot for the PPT slides
- [ ] Decide on team contribution split for the Contribution Table (still needed in notebook!)
- [ ] Re-run the notebook top-to-bottom in clean kernel to ensure all outputs are fresh
