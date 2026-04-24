# Credit Card Fraud Detection — What We Did & Why

> **Course:** Basics of AI and Machine Learning — Final Project
> **Presentation:** 16 April 2026
> **Dataset:** 339,607 credit card transactions, western United States

---

## The Starting Point

A fictional credit card company in the western US hired us to build a fraud detector. They gave us a spreadsheet of ~340,000 past transactions — each row is one purchase, with details like where it happened, when, how much, who the customer was, and whether it turned out to be fraud.

Our goal: teach a machine to look at a **new transaction** and decide — is this fraud or not?

---

## Why Standard Accuracy Doesn't Work Here

The first thing we noticed: only **0.52% of transactions are fraud** — 1,782 out of 339,607. The other 337,825 are completely normal.

This creates a trap. If a model just said *"everything is fine"* every single time, it would be **99.5% accurate** — but it would catch **zero fraud**. That's useless.

So we shifted focus to **Recall** — out of all real fraud cases, what percentage did we actually catch? The company also told us: *"It's OK to flag a few legitimate transactions by mistake — just don't miss real fraud."* That confirmed Recall as our number one metric.

---

## Step 1 — We Explored the Data First

Before building anything, we spent time understanding what fraud actually *looks like* in this dataset.

**What we found:**
- Fraud peaks between **1 AM and 4 AM** — most normal people aren't shopping at 3 AM
- Fraud transactions tend to be **larger amounts** — the average fraud transaction was $518, compared to ~$70 for legitimate ones
- Certain merchant types had much higher fraud rates — especially online shopping and grocery stores
- Fraudulent merchants were often **far from the cardholder's home address** — a card being used thousands of kilometres away is suspicious

This exploration didn't just produce charts. It told us *which signals to look for*, which directly shaped what we built next.

---

## Step 2 — We Created Better Features

The raw dataset had 15 columns — basic things like transaction time, amount, merchant name, and location coordinates. But the model can't do much with "lat: 39.7, merchant_lat: 48.8."

So we **engineered 6 new features** — calculated columns that turned raw data into meaningful signals:

| New feature | What it is | Why it matters |
|------------|------------|---------------|
| **Hour of day** | What time the transaction happened | Fraud clusters at night |
| **Day of week** | Weekday or weekend | Patterns differ |
| **Month** | Month of the year | Seasonal trends |
| **Cardholder age** | Age at time of purchase | Some age groups are targeted more |
| **Distance to merchant** | Straight-line distance from home to shop | A card used 9,000 km from home is a red flag |
| **Log of amount** | A rescaled version of the transaction amount | Raw dollar amounts have extreme outliers — log scaling makes them easier for models to use |

We also converted text categories (like merchant type and US state) into numbers, because machine learning models only understand numbers.

After all of this, we went from 15 raw columns to **39 features** the model could learn from.

---

## Step 3 — We Split the Data Carefully

We divided the dataset: **80% for training**, **20% locked away for testing**.

**Why?** The model learns patterns from the training set. The test set is data it has never seen — like a final exam. If we tested on the same data we trained on, the model could just "memorise" the answers and look great while failing on real new transactions.

We also used `stratify=y` — this ensures the 0.52% fraud rate is preserved in *both* sets. Without this, random chance could put almost all fraud cases into one half.

---

## Step 4 — We Fixed the Imbalance (SMOTE)

Even with 80% of the data, the training set still only had ~1,400 fraud cases out of ~271,000 rows. The model barely ever sees a fraud example — so it barely learns what fraud looks like.

We used **SMOTE** (Synthetic Minority Oversampling Technique) to generate realistic *synthetic* fraud rows and bring the training set to a 50/50 balance.

**Important:** we only did this to the **training data**. The test set stays 100% real and untouched — otherwise we'd be cheating on our own exam.

We also scaled all numeric features (amounts, distances, ages) to a similar range. Without this, a feature like "amount in dollars" (0–5,000) would mathematically overpower "hour of day" (0–23) just because it's a bigger number — even if hour is more useful.

---

## Step 5 — We Trained Five Different Models

We didn't just pick one algorithm. We trained five, to see which approach works best on this specific problem:

1. **Logistic Regression** — the simplest. Draws a straight line to separate fraud from legit
2. **Decision Tree** — asks a series of yes/no questions (like 20 questions)
3. **Random Forest** — builds hundreds of decision trees and takes a majority vote
4. **XGBoost** — each new tree corrects the mistakes of the previous one
5. **LightGBM** — same idea as XGBoost, but faster and slightly more accurate here

For each model, we automatically tested hundreds of different settings and kept the best version — a process called **hyperparameter tuning**.

---

## Step 6 — We Compared the Results

We tested all five models on the held-back 20% test set. Here's what we found:

| Model | Recall | F1-Score | ROC-AUC |
|-------|--------|----------|---------|
| LightGBM | 84.0% | 88.1% | 0.991 |
| XGBoost | 83.7% | 87.0% | 0.988 |
| Random Forest | 81.7% | 85.9% | 0.978 |
| Decision Tree | 80.1% | 80.1% | 0.900 |
| Logistic Regression | 77.5% | 76.3% | 0.877 |

**LightGBM won** — it caught the most fraud, with the fewest false alarms.

In business terms: compared to the worst model, LightGBM prevented an estimated **~$61,650 more in fraud losses**. Model selection isn't just a technical decision — it has a real financial impact.

---

## Step 7 — We Looked Inside the Model

After training, we asked LightGBM: *"which features did you rely on the most?"*

The top signals were: **Transaction Amount → Cardholder Age → Distance to Merchant → Time of Day → Merchant Location**

Four of the top five features were either created by us or location-based. This confirms that the feature engineering we did in Step 2 wasn't just busywork — it genuinely made the model better.

---

## Step 8 — We Tuned the Decision Threshold

By default, if a model is more than 50% confident something is fraud, it flags it. But that threshold is adjustable.

**Lowering it to 0.3** → the model becomes more cautious, catches more fraud (Recall: 86%), but raises more false alarms.

**Raising it to 0.7** → fewer false alarms, but misses more real fraud (Recall: 82%).

We found that **threshold 0.4** gives the best balance — Recall 85.4%, Precision 91.6%, F1-Score 88.4%. The company can adjust this depending on how risk-tolerant they want to be.

---

## Step 9 — We Tested Everything

Before submitting, we ran **19 automated tests** across 3 notebooks to verify:
- All model files load correctly
- Feature engineering math is correct (distance formula, age calculation, log scaling)
- The model gives consistent, sensible predictions

All 19 tests passed.

---

## What the Live Demo Shows

We deployed LightGBM as a web service. During the presentation, you can:
1. Generate a fake transaction
2. Send it to the model
3. See the fraud probability and verdict in real time

This proves the model works outside the notebook — not just on historical data.

---

## The Short Version (If Someone Asks)

> *"We had 340,000 credit card transactions and only 0.52% were fraud. We built new features from the raw data — especially distance between the cardholder and the merchant — trained five different machine learning models, and found that LightGBM caught 84% of all fraud cases while keeping false alarms low. In business terms, that's about $61,650 more in prevented fraud losses compared to the simplest model."*

---

## Key Terms

| Term | Plain English |
|------|--------------|
| **Recall** | Of all real fraud cases, what % did we catch? |
| **Precision** | Of all transactions we flagged, what % were actually fraud? |
| **F1-Score** | A single score balancing Recall and Precision |
| **ROC-AUC** | How well the model separates fraud from legit overall (1.0 = perfect) |
| **False Positive** | We said fraud — it was actually a real purchase. A false alarm |
| **False Negative** | We said legit — it was actually fraud. The dangerous miss |
| **SMOTE** | Creates synthetic fraud examples so the model sees enough to learn from |
| **Feature Engineering** | Building new useful columns from existing data |
| **Threshold** | The confidence cutoff above which the model calls something fraud |
| **Gradient Boosting** | Each tree corrects the previous tree's mistakes (LightGBM & XGBoost) |

---

## Where to Find Things

| What | Where |
|------|-------|
| Full analysis | `ML_work_and_analysis.ipynb` |
| Presentation slides | `Presentation/Presentation PPT.html` |
| Speaker notes | `Presentation/Speech Notes.html` |
| Tests | `Phase_1_testing.ipynb`, `Phase_2_testing.ipynb`, `Phase_3_testing.ipynb` |
| Live demo | `docker-compose up --build` → open `localhost:3000` |
