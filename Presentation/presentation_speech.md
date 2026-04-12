# Presentation Speech — Credit Card Fraud Detection with Machine Learning

> **Course:** Basics of AI and Machine Learning — Final Group Project
> **Date:** 16 April 2026
> **Duration:** ~12–15 minutes (your part, excluding live demo)
> **Note:** `[SLIDE X]` markers tell you when to advance. Speak naturally — these are talking points, not a script to read word for word.

---

## [SLIDE 1] Title Slide

Good afternoon everyone. Our project is **Credit Card Fraud Detection using Machine Learning**. Today we'll walk you through our full pipeline — from raw transaction data to a trained model that can flag fraud in real time.

---

## [SLIDE 2] The Problem

Imagine you're a new credit card company in the western United States. You've just launched, and you're marketing yourself as one of the safest cards to use. Your reputation depends on one thing: **catching fraud before it costs your customers money.**

The executive who hired us was very clear — the model should **err on the side of caution**. It's better to flag a legitimate transaction and inconvenience a customer for a moment than to let real fraud slip through. So our primary goal is **high Recall** — out of all actual fraud cases, how many does the model catch?

---

## [SLIDE 3] The Dataset

We worked with a Kaggle dataset of **339,607 credit card transactions** from the western US. Each row is one purchase — it includes who bought what, where, when, how much, and crucially, whether or not it was fraud.

Here's the core challenge: only **0.52% of all transactions are fraud**. That's 1,761 fraud cases out of nearly 340,000 transactions. Think of it like finding 5 bad apples in a crate of 1,000.

This means if we built a model that always says "not fraud" — every single time — it would technically be **99.5% accurate**. But it would catch **zero** fraud. So accuracy is a completely misleading metric here. We need something better, and that's where Recall, F1-Score, and ROC-AUC come in.

---

## [SLIDE 4] EDA Highlights

Before training any models, we spent time exploring the data visually. This is called **Exploratory Data Analysis**, or EDA. The goal is to understand what fraud actually looks like in the data.

Here are our key discoveries:

**First — timing.** Fraud peaks between **1 and 4 AM**. That makes intuitive sense — if someone steals your card, they're more likely to use it while you're asleep and won't notice right away.

**Second — amounts.** Fraudulent transactions have a **significantly higher median amount** than legitimate ones. Criminals tend to go for bigger purchases.

**Third — merchant categories.** Categories like `grocery_pos`, `shopping_net`, and `misc_net` had the highest fraud rates. Online shopping categories appear more frequently in fraud, which makes sense — there's no physical card verification.

**And fourth — geography.** The distance between the cardholder's location and the merchant turned out to be a very strong signal. If someone in California has a transaction at a merchant 8,000 km away, that's suspicious.

These patterns directly guided our feature engineering in the next step.

---

## [SLIDE 5] Feature Engineering

This is one of the most important parts of the project. The raw dataset had columns like timestamps and coordinates, but the model can't directly learn from a datetime string. So we **engineered 6 new features**:

1. **`hour`** — extracted from the transaction timestamp. Captures the night-time fraud pattern we found in EDA.
2. **`day_of_week`** and **`month`** — also from the timestamp. Captures weekly and seasonal patterns.
3. **`age`** — calculated from the customer's date of birth minus the transaction date.
4. **`distance_km`** — calculated using the **Haversine formula**, which gives us the straight-line distance in kilometres between the cardholder's coordinates and the merchant's coordinates. This ended up being one of the top features.
5. **`log_amt`** — the natural logarithm of the transaction amount. This compresses the extreme range of values — amounts go from $1 to over $28,000 — and reduces the influence of outliers.

We also **one-hot encoded** the categorical columns — `category` with 14 values and `state` with 13 values — which converts them into binary 0/1 columns that the model can process.

And we dropped columns that wouldn't help prediction — things like merchant name, city, job title, and the transaction ID.

---

## [SLIDE 6] Data Preparation

Three critical preprocessing steps:

**Train-test split:** We split the data **80% for training, 20% for testing**, using `stratify=y` to maintain the same fraud ratio in both sets, and `random_state=42` for reproducibility.

**Feature scaling:** We applied **StandardScaler** to the numeric features. This rescales them to have a mean of 0 and standard deviation of 1. This matters especially for Logistic Regression, which is sensitive to feature magnitudes. Importantly, we fitted the scaler **only on training data** and then applied it to both sets — this prevents **data leakage**, which would artificially inflate our metrics.

**SMOTE:** This stands for **Synthetic Minority Over-sampling Technique**. Remember, only 0.52% of our training data was fraud. SMOTE creates **synthetic fraud examples** by interpolating between real fraud cases using nearest neighbours. After SMOTE, our training set was balanced at 50/50. The key rule: SMOTE is **only applied to training data** — the test set stays untouched, because it needs to reflect the real-world distribution.

---

## [SLIDE 7] Models Trained

We trained **5 different models**, starting simple and increasing in complexity:

**1. Logistic Regression** — our baseline. It draws a single linear boundary and outputs a probability. It's fast, interpretable, but limited — it can't capture non-linear feature interactions.

**2. Decision Tree** — learns a series of yes/no questions. "Is the amount greater than $500? Is the hour past midnight?" It can capture non-linearity but tends to **overfit** — memorise the training data rather than generalise.

**3. Random Forest** — an **ensemble** of many decision trees, each trained on a random subset of data and features. The trees vote together. This reduces overfitting and gives more stable predictions. Think of it as crowd wisdom — the crowd is wiser than any single individual.

**4. XGBoost** — **Extreme Gradient Boosting**. Instead of training independent trees, each new tree specifically tries to correct the mistakes of the previous one. It's iterative — like having a team of editors, each one fixing what the last one missed.

**5. LightGBM** — **Light Gradient Boosting Machine**, developed by Microsoft. Same boosting idea as XGBoost, but faster and more memory-efficient on large datasets.

For all models, we used **5-fold cross-validation** during hyperparameter tuning. We tuned using **GridSearchCV** for simpler models and **RandomizedSearchCV** for the more complex ones with larger search spaces. The scoring metric was **F1-Score**, which balances Recall and Precision.

---

## [SLIDE 8] Results Comparison

Here are the actual results on the held-out test set — data the models never saw during training:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **LightGBM** | 0.9988 | 0.9257 | 0.8399 | 0.8807 | 0.9978 |
| **XGBoost** | 0.9987 | 0.8691 | 0.8764 | 0.8727 | 0.9964 |
| **Random Forest** | 0.9978 | 0.8475 | 0.7022 | 0.7680 | 0.9903 |
| **Decision Tree** | 0.9965 | 0.6343 | 0.7697 | 0.6954 | 0.8837 |
| **Logistic Regression** | 0.9888 | 0.2699 | 0.6657 | 0.3841 | 0.8767 |

A few things to notice:

**Accuracy is nearly identical across all models** — ranging from 98.9% to 99.9%. This perfectly illustrates why accuracy is useless on imbalanced data. The real story is in Recall and F1.

**The gradient boosting models clearly dominate.** LightGBM and XGBoost both achieve F1-Scores around 0.87–0.88 — a huge jump from Logistic Regression's 0.38.

**XGBoost has the highest Recall at 0.8764** — it catches 312 out of 356 fraud cases. LightGBM catches 299 but has higher Precision, meaning fewer false alarms — only 24 compared to XGBoost's 47.

**Logistic Regression catches only 237 of 356 fraud cases** — that's 62 more missed fraud cases compared to XGBoost.

---

## [SLIDE 9] ROC and Precision-Recall Curves

The **ROC curve** plots True Positive Rate against False Positive Rate at every possible threshold. A perfect model hugs the top-left corner. All our models have AUC above 0.87, but this can be misleading on imbalanced data because the huge number of legitimate transactions inflates the True Negative Rate.

That's why we also plotted the **Precision-Recall curve**, which is more honest on imbalanced datasets. It focuses only on the positive class — fraud. Here you can clearly see the gradient boosting models pulling away from the simpler ones. The area under this curve is called **Average Precision**, and it gives a more realistic picture of how well each model performs on the rare fraud class.

---

## [SLIDE 10] Confusion Matrices

The confusion matrix breaks down predictions into four categories:

- **True Positives** — fraud correctly caught
- **True Negatives** — legitimate correctly cleared
- **False Positives** — legitimate flagged as fraud (annoying, but manageable)
- **False Negatives** — fraud missed (the **most dangerous** error)

Comparing the best and worst models: the best model has far fewer False Negatives in the bottom-left cell. Each number in that cell represents a real fraud case that slipped through undetected.

---

## [SLIDE 11] Business Impact

To make this tangible, we calculated the financial impact.

The **average fraud transaction value** in our dataset is **$518.07**.

Our best-performing models caught 299–312 out of 356 fraud cases. Logistic Regression caught only 237. That's a difference of **62–75 fraud cases**.

In dollar terms, choosing the best model over the worst **prevents approximately $32,000–$39,000 in estimated fraud losses** on this test set alone.

Scale that up to millions of real transactions per year and the difference becomes enormous. This is why model selection is not just a technical exercise — it's a **business decision**.

---

## [SLIDE 12] Threshold Optimisation

By default, we use a threshold of 0.5 — if the model's predicted fraud probability is 50% or higher, we flag it as fraud.

But that threshold is **tuneable**. We tested our best model at thresholds from 0.1 to 0.7:

- At **threshold 0.3**, Recall jumps to 0.87 — we catch more fraud, but get more false alarms
- At **threshold 0.5** (default), Recall is 0.84, Precision is 0.93 — a strong balance
- At **threshold 0.7**, Precision rises to 0.94 but Recall drops to 0.82

The **optimal threshold for F1 was 0.4**, giving Recall of 0.854, Precision of 0.916, and F1 of 0.884.

This is a practical insight: depending on how cautious the business wants to be, you can turn this dial. A bank that prioritises catching every possible fraud would lower the threshold. One that wants fewer customer interruptions would raise it.

---

## [SLIDE 13] Feature Importance

After training, we extracted the top 25 most important features from our best model. Here's what the model actually learned to rely on:

1. **`amt` / `log_amt`** — transaction amount is the single strongest signal
2. **`distance_km`** — how far the merchant is from the cardholder
3. **`hour`** — time of day, reflecting the night-time fraud pattern
4. **`age`** — age of the cardholder
5. **Certain merchant categories** — particularly online shopping and miscellaneous

The important thing is: **our engineered features ranked among the top.** `distance_km`, `hour`, `log_amt`, and `age` were all created by us during feature engineering. The raw dataset didn't contain them. This validates that **domain knowledge translated into real predictive power**.

---

## [SLIDE 14] Why Gradient Boosting Wins

A quick explanation of why the model hierarchy looks the way it does:

- **Logistic Regression** draws a single straight line between fraud and non-fraud. It cannot capture combinations like "high amount + night-time + online category + far away." It's too simple for this problem.

- **Decision Tree** can capture non-linear patterns, but a single tree overfits — it memorises training noise instead of learning generalisable patterns.

- **Random Forest** averages many independent trees, which stabilises predictions. But the trees don't learn from each other's mistakes.

- **XGBoost and LightGBM** use **gradient boosting** — each new tree is specifically trained to correct the errors of all previous trees. This iterative error-correction is what allows them to capture the subtle multi-feature fraud patterns that other models miss.

---

## [SLIDE 15] Testing and Validation

To make sure everything works correctly, we built a **3-phase test suite** across three separate notebooks:

- **Phase 1** (5 tests) — verifies that all saved model artefacts load correctly: the LightGBM model, the scaler, the feature column list, and the numeric feature list.
- **Phase 2** (9 tests) — validates the feature engineering math: Haversine distance calculation, log transformation, age computation, hour/day/month extraction, one-hot encoding alignment, and StandardScaler application.
- **Phase 3** (5 tests) — tests model inference end-to-end: valid class labels, probabilities between 0 and 1, deterministic outputs across repeated calls, directional sanity (forced fraud patterns score higher than normal), and batch prediction on 5 mixed transactions.

**All 19 tests pass.** This gives us confidence that the pipeline is correct from feature engineering through to prediction.

---

## [SLIDE 16] Live Demo

Now I'll hand over to [colleague's name] for the live demonstration. They'll show the model running in real time — generating synthetic transactions and scoring them through our Flask API and Next.js dashboard.

*[Colleague takes over for demo]*

---

## [SLIDE 17] Limitations and Future Work

A few things we'd acknowledge:

**Limitations:**
- We used a **random train/test split**, not a temporal one. In real life, fraud patterns change over time — a time-based split would better simulate how the model performs on future, unseen data.
- Our **distance feature** uses straight-line Haversine distance, not actual road distance.
- This is a **single static dataset** — a production system would need continuous retraining as fraud patterns evolve.

**What we'd explore with more time:**
- **SHAP values** — to explain *why* a specific transaction was flagged, not just whether it was. This matters for compliance and auditing.
- **Time-series cross-validation** — respecting the chronological order of transactions.
- **Cost-sensitive learning** — building the asymmetric cost of missed fraud directly into the loss function, rather than handling it through SMOTE.
- **Deep learning architectures** like TabNet or FT-Transformer that have recently shown competitive results on tabular data.

---

## [SLIDE 18] Key Takeaways and Q&A

To summarise our five main takeaways:

1. **Accuracy is meaningless on imbalanced data.** Use Recall, F1, and ROC-AUC instead.
2. **Feature engineering matters as much as model choice.** Our hand-crafted features — distance, hour, log amount, age — ranked among the top predictors.
3. **Gradient boosting dominates tabular fraud detection.** XGBoost and LightGBM consistently outperformed simpler models.
4. **Preventing data leakage is critical.** SMOTE on training only, scaler fitted on training only.
5. **Model selection is a business decision.** The difference between the best and worst model translates to tens of thousands of dollars in missed fraud.

Thank you. We're happy to take questions.

---

## Quick Reference — If You Get a Question

**"Why not deep learning?"**
> Gradient boosting consistently outperforms deep learning on tabular data. There's even a 2021 paper on this — Shwartz-Ziv & Armon showed that XGBoost beats deep models on most tabular benchmarks. Deep learning shines on images, text, and sequences — not spreadsheets.

**"What is overfitting?"**
> When a model memorises the training data instead of learning general patterns. It performs great on training data but poorly on new data. We combat this with cross-validation, regularisation, and ensemble methods.

**"Why SMOTE and not just class weights?"**
> Both are valid. SMOTE creates synthetic examples, which gives the model more diverse training signal. Class weights just change how much each mistake costs. We chose SMOTE because it's the most commonly referenced technique in the literature for this type of problem and produces good results.

**"Could this work in real time?"**
> Yes — our demo shows exactly that. The LightGBM inference takes milliseconds. In a production system, you'd wrap it in a streaming pipeline and score each transaction as it arrives.
