# Presentation Speech — Credit Card Fraud Detection with Machine Learning

> **Course:** Basics of AI and Machine Learning — Final Group Project
> **Date:** 16 April 2026
> **Duration:** ~12–15 minutes (your part, excluding live demo)
> **Note:** `[SLIDE X]` markers tell you when to advance. Speak naturally — these are talking points, not a script to read word for word.

---

## [SLIDE 1] Title Slide

Good afternoon, everyone. My name is Jolanta, and today I'll be presenting our final project — **Credit Card Fraud Detection using Machine Learning**.

What we built is a full end-to-end machine learning pipeline. That means we started with raw transaction data — just numbers in a spreadsheet — and ended up with a trained model that can tell you, in milliseconds, whether a transaction looks fraudulent or not. My colleague will show you that model running live in a moment. But first, let me walk you through how we got there, and why the decisions we made along the way actually matter.

---

## [SLIDE 2] The Problem — Why Fraud Detection Is Hard

Let me start with the problem we were solving.

The project is built around a realistic business scenario. A new credit card company has just launched in the western United States. They're marketing themselves as one of the safest cards available — and their reputation depends entirely on one thing: catching fraud before it costs their customers money.

**The brief.**

Our main goal was simple to state: catch as much fraud as possible, before the customer loses money. The instruction we were given was: **when in doubt, stop the transaction**. A blocked card can be unblocked in minutes. Money stolen is gone.

**The accuracy trap.**

Now — before we even looked at a single model, we ran into a fundamental problem. Only **0.52% of all transactions in the dataset are fraud**. That's 1,782 fraud cases out of nearly 340,000 transactions.

Think about what that means. If you built the laziest possible model — one that looks at every transaction and always says "not fraud, approve it" — that model would be **99.5% accurate**. Sounds impressive. But it catches absolutely zero fraud. Every single criminal walks away.

So accuracy is a completely misleading number here. We needed different metrics — ones that focus specifically on what happens to the fraud cases, not just the overall score. I'll explain exactly which metrics we used, and why, when we get to the results section.

**Two types of mistakes.**

Before we get there, I want to establish one more idea — because it shapes everything about how we built this.

In fraud detection, there are two ways the model can be wrong:

The first: the model blocks a real, legitimate purchase by mistake. The customer is annoyed, calls the bank, and the card is unblocked within minutes. Inconvenient — but fixable. No lasting damage.

The second: the model misses actual fraud. The criminal goes through. The customer loses money. The bank pays the chargeback. Trust is damaged — and that money is gone.

These two mistakes are not equal. One costs a phone call. The other costs real money and real trust. And that asymmetry is why every decision we made was tilted in one direction — catch more fraud, even if it means occasionally stopping a real purchase.

---

## [SLIDE 3] The Dataset

Let me show you the data we worked with — because the dataset itself is a core part of the story.

We used a Kaggle dataset of **339,607 credit card transactions** from the western United States. Each row represents one purchase. The raw data gave us 15 columns — who the cardholder is, their date of birth, home address and coordinates, the merchant name, the merchant's location and coordinates, what category the purchase belongs to, the transaction amount, and a precise timestamp. And one more column: **is_fraud** — a simple 1 or 0 that tells us whether the transaction was later confirmed as fraud.

So the data is rich. We have geography, timing, amounts, categories, customer demographics. Everything a fraud analyst would want to look at.

Now here's where it gets difficult. Out of those 339,607 transactions, only **1,782 are fraud**. That's **0.52%**. Not even one percent.

*[gesture toward the charts]*

The bar chart shows this visually — and it's almost impossible to see. The fraud bar is barely a sliver next to 337,825 legitimate transactions. The pie chart shows the percentages: 99.48% legitimate, 0.52% fraud.

Here's a way to think about it that makes the scale concrete. Imagine you're screening people at an airport. For every 200 passengers who walk through, one of them is carrying something they shouldn't. The other 199 are completely innocent. Your job is to find that one person without holding up the other 199 for too long.

That's the problem. And it immediately tells us something important about what will go wrong if we're not careful.

**A model that always says "not fraud" — for every single transaction, no exceptions — would score 99.5% accuracy.** It's not wrong very often! But it's completely useless. It catches zero fraud. Not one criminal is stopped.

This is the accuracy trap I mentioned in the previous slide, showing up in the data. And that imbalance doesn't just affect how we measure things — it shapes every decision that follows: how we train the model, how we balance the data before training, and which metrics we use to evaluate performance. Everything you'll see in the next sections flows from this single fact about the dataset.

---

## [SLIDE 4] EDA Highlights

Before we touched any models, we spent time just looking at the data. This stage is called **Exploratory Data Analysis** — EDA for short — and the goal is simple: understand what fraud actually looks like before you try to teach a machine to detect it.

We found four patterns that turned out to be really important.

**The first one is timing.** Fraud doesn't happen randomly throughout the day. It peaks sharply between **1 and 4 in the morning**. Think about why — if someone steals your card details, they want to use them before you notice. And the best time to do that is while you're asleep and not checking your bank app.

**The second pattern is amounts.** Fraudulent transactions are significantly larger than legitimate ones. The average fraudulent transaction in this dataset is **$518**. The average legitimate transaction is around **$70**. That's nearly an eight-to-one difference. Criminals go for bigger purchases — they're not buying coffee.

**Third — merchant categories.** The types of merchants with the highest fraud rates were online categories — online shopping, online miscellaneous. This makes intuitive sense: online purchases don't require the physical card to be present, so stolen card numbers are easy to use there. No cashier, no signature, no chip to read.

**And fourth — geography.** We calculated the distance between where the cardholder lives and where the merchant is located. Fraudulent transactions tend to happen at merchants that are much further away from the cardholder's home. A purchase at a merchant thousands of kilometres away, on a card registered in California — that's a very suspicious signal. In fact, distance ended up as the **third most important feature** in the final model — behind transaction amount and cardholder age, both of which also ranked extremely high.

You can think of these four patterns as one fingerprint — hour, amount, distance, category. They don't each prove fraud on their own. But together, they paint a picture. And every single one translated directly into a feature we engineered in the next step.

---

## [SLIDE 5] Feature Engineering

This is probably the part of the project I'm most proud of, because it's where domain knowledge and engineering work together.

The raw dataset gave us things like a timestamp column and latitude/longitude coordinates. But you can't feed a timestamp into a machine learning model and expect it to learn anything useful. Models need numbers. So we had to transform the raw data into something the model could actually reason about.

We created **6 new features**:

The first three all come from the timestamp: **`hour`**, **`day_of_week`**, and **`month`**. Extracting the hour is what allows the model to learn that 3 AM transactions are more suspicious than 3 PM transactions. Without this feature, the model would never see that pattern — the raw timestamp is just a string.

The fourth feature is **`age`** — calculated by subtracting the cardholder's date of birth from the transaction date. Fraud rates vary significantly across age groups, and this gave the model a signal it couldn't get from the raw data.

The fifth feature — and this one became one of our most important — is **`distance_km`**. We used something called the **Haversine formula**, which calculates the straight-line distance in kilometres between two points on the Earth's surface. We applied it to the cardholder's home coordinates and the merchant's coordinates. A large distance is a strong red flag, and the model learned to use it.

The sixth feature is **`log_amt`** — the natural logarithm of the transaction amount. Transaction amounts in this dataset range from under a dollar to over $28,000. That enormous range causes problems for some models. Taking the log compresses it into a much smaller, more manageable range, and reduces the distorting influence of extreme outliers.

We also one-hot encoded the categorical columns — `category` with 14 values, `state` with 13 — converting them into binary columns the model can process. And we dropped columns that carry no predictive value: merchant name, job title, city, transaction ID. Those are too unique or too arbitrary to generalise.

The result: **15 raw columns became 39 engineered features**. And when we later looked at feature importance, four of our hand-crafted features ranked in the top predictors. The model learned what we taught it to look for.

---

## [SLIDE 6] Data Preparation

Before any model sees the data, three critical steps have to happen — and getting these wrong is one of the most common ways to produce results that look great in a notebook but fail in production.

**Step one: Train-test split.** We split the data 80% for training and 20% for testing. The 20% test set is locked away and never touched until the very end — it's the real exam. We used `stratify=y` to make sure both sets have the same fraud ratio, and `random_state=42` for reproducibility — so anyone running our code gets the same results.

**Step two: Feature scaling.** We applied **StandardScaler** to the numeric features. This rescales them to have a mean of zero and a standard deviation of one. Why does this matter? Some models — particularly Logistic Regression — are sensitive to the magnitude of features. If one feature ranges from 0 to 28,000 and another from 18 to 90, the first one will dominate just because of its scale, not because it's more informative. Scaling puts everyone on equal footing.

Critically — and this is important — we fitted the scaler **only on the training data**. Then we applied it to both training and test. If we had fitted it on all the data at once, the test set information would have leaked into the training process. That's called **data leakage**, and it leads to metrics that look better than they really are. We were very careful to avoid it.

**Step three: SMOTE.** This stands for **Synthetic Minority Over-sampling Technique**. Remember our 0.52% fraud rate? If we train a model on data where 99.5% of examples say "not fraud", the model will learn that "not fraud" is almost always the right answer. It barely sees enough fraud examples to learn what fraud looks like.

SMOTE solves this by creating **synthetic fraud examples**. It looks at real fraud cases, finds their nearest neighbours in feature space, and generates new examples that interpolate between them. After SMOTE, our training set was balanced at 50/50 — the model sees equal amounts of fraud and legitimate transactions.

The golden rule: SMOTE is **only applied to the training set**. The test set stays completely untouched — because it needs to reflect the real-world distribution we're trying to predict.

---

## [SLIDE 7] Models Trained

We trained five models, deliberately starting simple and increasing in complexity. This isn't random — it's the standard scientific approach. You establish a baseline first, then you know exactly how much improvement each additional step buys you.

**Model 1: Logistic Regression.** Our baseline. It draws a single straight-line boundary between fraud and not-fraud, and outputs a probability. It's fast, interpretable, easy to explain. But it has a fundamental limitation — it assumes the relationship between features and the outcome is linear, which means it can't capture interactions like "high amount AND night-time AND far away." It either/or thinking.

**Model 2: Decision Tree.** This learns a series of yes/no questions — "Is the amount over $500? Is the hour after midnight? Is the distance over 1,000 km?" It can capture those non-linear interactions the Logistic Regression misses. But a single decision tree has a tendency to **overfit** — it memorises the training data, including the noise, and doesn't generalise well to new transactions it hasn't seen before.

**Model 3: Random Forest.** The answer to overfitting. Instead of one tree, we grow hundreds of them — each trained on a random subset of the data and a random subset of the features. Then we let them vote. No single tree dominates; the majority rules. Think of it as crowd wisdom — a crowd of 500 people, each with slightly different information, is more reliable than any single expert. Random Forest is a big step up from Decision Tree.

**Model 4: XGBoost — Extreme Gradient Boosting.** Now we move from parallel trees to sequential ones. XGBoost builds trees one at a time, where each new tree specifically focuses on correcting the mistakes the previous trees made. It's like a team of editors where each person only fixes what the previous person got wrong. Iterative, focused, powerful.

**Model 5: LightGBM — Light Gradient Boosting Machine**, developed by Microsoft. Same sequential boosting idea as XGBoost, but engineered for speed and memory efficiency on large datasets. It's the model we ultimately deployed — chosen not because it catches the absolute most fraud, but because it catches nearly as much as XGBoost **while producing far fewer false alarms**. 24 false positives versus 47 for XGBoost. That means fewer legitimate customers being blocked unnecessarily. I'll come back to that trade-off when we look at the business impact.

Every model was tuned using **5-fold cross-validation** — we split the training data into 5 parts, trained on 4 and validated on 1, rotated through all 5 combinations, and averaged the results. This gives a much more reliable estimate of performance than a single validation run. We optimised on **F1-Score**, which balances Recall and Precision.

---

## [SLIDE 8] Results Comparison

The three metrics — Recall, F1-Score, and ROC-AUC — are defined on the Models slide you just saw. So let me go straight to what the numbers actually show.

These are results on the held-out test set. Data none of the models ever saw during training.

*[point to table]*

A few things jump out immediately.

Look at the Accuracy column first. It barely moves — 98.9% to 99.9% across all five models. This is exactly what we predicted. Accuracy is useless here. Every model, even the worst one, looks "accurate" because the dataset is 99.5% legitimate transactions.

Now look at Recall and F1. **That's** where you see the real differences.

Logistic Regression — our baseline — has a Recall of 0.6657. That means it misses about one in three fraud cases. F1 of 0.38. That's not good enough.

Random Forest improves significantly — Recall of 0.70, F1 of 0.77.

And then the gradient boosting models pull ahead clearly. **XGBoost reaches Recall 0.8764** — that means it catches 312 out of 356 fraud cases in the test set. **LightGBM reaches Recall 0.8399**, catching 299 cases, but with Precision of 0.9257 — meaning far fewer false alarms. Only 24 false positives versus XGBoost's 47.

Both are excellent. XGBoost catches slightly more fraud. LightGBM bothers fewer legitimate customers. The right choice depends on the business's priorities.

---

## [SLIDE 9] ROC and Precision-Recall Curves

The **ROC curve** — Receiver Operating Characteristic — plots True Positive Rate against False Positive Rate at every possible decision threshold. A perfect model hugs the top-left corner. All our models score above 0.87 AUC, which sounds great — but ROC-AUC can be misleading on imbalanced datasets, because the enormous number of legitimate transactions inflates the True Negative Rate.

That's why we also look at the **Precision-Recall curve**. This curve only cares about the fraud class — it completely ignores the easy cases. On this curve, you can clearly see the gradient boosting models pulling away from the simpler ones. The area under this curve is called **Average Precision**, and it's a much more honest picture of how well a model handles the rare fraud class.

---

## [SLIDE 10] Confusion Matrices

The confusion matrix is the clearest way to see what a model actually does. It breaks every prediction into one of four boxes.

**True Positives** — fraud correctly caught. The model did its job.
**True Negatives** — real purchases correctly approved. The easy majority.
**False Positives** — a real purchase was blocked by mistake. The customer is annoyed, calls the bank, card is unblocked. Manageable.
**False Negatives** — real fraud slipped through undetected. The customer loses money. This is the number we are most focused on reducing.

When you compare the best model to the worst, the difference in False Negatives is dramatic. Logistic Regression has 119 False Negatives. XGBoost has 44. That's 75 fraud cases that either got caught or didn't, depending purely on which model we chose.

---

## [SLIDE 11] Business Impact

Let me translate that into something concrete.

The **average fraudulent transaction** in our dataset is **$518.07**. That's the average amount a fraudster spends when they get through.

Our best models — XGBoost and LightGBM — caught between 299 and 312 of the 356 fraud cases in the test set. Logistic Regression caught only 237.

The difference between the best and worst model is **62 to 75 fraud cases**. Multiply that by $518 per transaction, and choosing the right model **prevents approximately $32,000 to $39,000 in estimated fraud losses** on this test set alone.

Now scale that up. A real credit card company processes millions of transactions every year. The financial difference between a well-chosen model and a naive one isn't thousands of dollars — it's potentially millions. And beyond the money, there's customer trust, regulatory compliance, and brand reputation. Model selection is not a technical exercise. It is a business decision.

---

## [SLIDE 12] Threshold Optimisation

One more practical insight that often gets overlooked.

When a model outputs a fraud prediction, it's actually outputting a **probability** — something like 0.73, meaning "I'm 73% confident this is fraud." We then apply a threshold to convert that into a binary yes/no decision. The default threshold is 0.5.

But 0.5 is arbitrary. We can move that dial.

We tested our best model at thresholds from 0.1 all the way to 0.7 and watched what happened to Recall and Precision.

At **threshold 0.3**, Recall climbs to 0.86 — we catch more fraud, but we also flag more legitimate transactions. More false alarms.

At **threshold 0.5** (the default), Recall is 0.84, Precision is 0.93. A strong balance.

At **threshold 0.7**, Precision rises to 0.94 — we're very confident when we flag something — but Recall drops to 0.82. We're catching less fraud.

The **optimal threshold for F1-Score was 0.4**, giving Recall 0.854, Precision 0.916, and F1 of 0.884.

This matters because different businesses have different priorities. A bank that cannot afford to miss a single fraud case lowers the threshold. A bank that prioritises customer experience and wants fewer interruptions raises it. The model doesn't make this decision — the business does.

---

## [SLIDE 13] Feature Importance

After training, we extracted which features the model relied on most. The top predictors were:

1. **`amt` and `log_amt`** — transaction amount. The single strongest signal, by a significant margin.
2. **`distance_km`** — how far the merchant is from the cardholder's home. One of the features we engineered.
3. **`hour`** — time of day. Another engineered feature, capturing the night-time pattern.
4. **`age`** — cardholder age, also engineered by us.
5. **Merchant categories** — particularly online shopping and miscellaneous online.

Here's what this tells us: four of the top features didn't exist in the raw data. We created them. The model learned from what we taught it to look for. That's what feature engineering is — translating human intuition about how fraud works into numbers a model can use. And in this case, it paid off directly in predictive performance.

---

## [SLIDE 14] Why Gradient Boosting Wins

Just to give you a clear picture of why the model rankings look the way they do.

Logistic Regression draws a single straight boundary. Fraud patterns are not linear — they involve combinations like "high amount AND late at night AND far away from home AND online category." A straight line can't capture that. Too simple.

Decision Tree can capture those combinations. But a single tree memorises the training data — it learns the noise along with the signal, and that noise doesn't generalise to new transactions.

Random Forest fixes the overfitting problem by averaging hundreds of trees. Much more stable. But the trees are independent — they don't learn from each other's mistakes.

XGBoost and LightGBM do something fundamentally different. Each new tree is built specifically to fix the errors of all the trees before it. Every iteration, the model becomes a little better at the cases it previously got wrong. That iterative error-correction is what lets it pick up the subtle, multi-feature patterns that slip past every other approach.

It's the difference between a committee where everyone votes independently, and a team where each person specifically addresses what the previous person missed.

---

## [SLIDE 15] Testing and Validation

We also built a formal test suite to verify that every part of the pipeline works correctly. This is something that separates a production-ready system from a notebook experiment.

We organised it into **three phases**, covering 19 tests in total.

**Phase 1** verifies that all saved model artefacts load correctly — the LightGBM model file, the fitted scaler, the feature column list, and the list of numeric features. If any of these fail, the API breaks before a single prediction is made.

**Phase 2** validates the feature engineering mathematics. We test that the Haversine distance formula gives correct results, that the log transformation is applied correctly, that age is computed from the right date fields, that hour, day, and month extraction works, and that one-hot encoding aligns to the expected column structure.

**Phase 3** tests end-to-end inference. We verify that outputs are valid class labels, that probabilities are between 0 and 1, that repeated calls on the same input give the same result, that a transaction with forced fraud characteristics scores higher than a normal one, and that batch prediction across 5 mixed transactions works correctly.

**All 19 tests pass.** This isn't just a formality — it's confidence that when my colleague runs the demo in a moment, the pipeline will behave exactly the way we designed it to.

---

## [SLIDE 16] Live Demo

And on that note — I'll hand over to [colleague's name] for the live demonstration. They'll show the model running in real time: generating synthetic transactions, running them through the feature engineering pipeline, scoring them against the LightGBM model, and displaying the fraud verdict live in the dashboard.

*[Colleague takes over for demo]*

---

## [SLIDE 17] Limitations and Future Work

Every honest presentation acknowledges what could be better. Here are ours.

**On the limitations side:**

Our train-test split was random — we shuffled the data and took 20% as the test set. In a real fraud detection system, that's not quite right. Fraud patterns change over time — criminals adapt their tactics as banks improve detection. A **time-based split** — where you train on older data and test on more recent data — would better simulate how the model would actually perform on future transactions it hasn't seen.

Our distance feature uses **straight-line Haversine distance**, not actual travel distance. Someone could be 500 km away as the crow flies but easily reachable by plane. A more nuanced geographic feature might capture this better.

And this is a **single, static dataset**. A production system would need continuous retraining as fraud patterns evolve — what looks like fraud today might look different in six months.

**Given more time, we'd explore:**

**SHAP values** — a technique that explains not just whether a transaction was flagged, but exactly which features drove that decision, and by how much. This matters for compliance and for building trust in the system.

**Time-series cross-validation**, which respects the chronological order of transactions rather than treating them as interchangeable.

**Cost-sensitive learning** — rather than rebalancing the data with SMOTE, you can directly tell the model that missing a fraud case costs ten times more than a false alarm. This bakes the business priority directly into the loss function.

And **deep learning architectures** like TabNet or FT-Transformer, which have shown promising results on tabular data in recent research.

---

## [SLIDE 18] Key Takeaways and Q&A

I want to close with five things you should take away from this project. These are on screen — I'll walk through each one briefly.

**One — accuracy is meaningless on imbalanced data.** When 99.5% of your data is one class, any model that ignores the minority looks great. Accuracy told us nothing. Recall, F1, and ROC-AUC told us everything. If you ever work on a classification problem with unbalanced classes, this is the first thing to remember.

**Two — feature engineering matters as much as model choice.** Hour, distance, age, and log amount — features we built from scratch — ranked among the top predictors. We didn't find those patterns by trying more algorithms. We found them by understanding the problem first, then building the right numbers.

**Three — gradient boosting dominates tabular fraud detection.** XGBoost and LightGBM outperformed every simpler model by a clear margin. Each tree in the sequence learns from the previous one's mistakes — that iterative focus is what captures subtle multi-feature fraud patterns that a single tree or a straight line never could.

**Four — data leakage is the most dangerous silent mistake.** SMOTE only on training data. Scaler fitted only on training data. Apply these to all data first and your metrics look better in the notebook — then fall apart in production. We were careful to avoid it, and that's exactly the kind of care that separates a working system from one that only works on paper.

**Five — model selection is a business decision, not a technical one.** The difference between our best and worst model is approximately $39,000 in prevented fraud on this test set alone. Scale that to millions of real transactions and the stakes are enormous. The people choosing the model need to understand what Recall actually means — and why it matters more than accuracy.

Thank you. We're happy to take questions.

---

## Quick Reference — If You Get a Question

**"Why not deep learning?"**
> Gradient boosting consistently outperforms deep learning on tabular data — this has been studied and replicated. There's a well-cited 2021 paper by Shwartz-Ziv and Armon that benchmarked this across many datasets and found XGBoost wins on tabular benchmarks. Deep learning shines on images, text, audio, and sequences — not on structured spreadsheet data like ours.

**"What is overfitting?"**
> When a model learns the training data too well — it memorises the noise along with the patterns. It performs beautifully on data it's already seen, and badly on new data. You can spot it when training accuracy is much higher than test accuracy. We combat it with cross-validation, regularisation, and ensemble methods like Random Forest.

**"Why SMOTE and not just class weights?"**
> Both are valid approaches, and we chose SMOTE because it gives the model more diverse training signal — it sees many different synthetic fraud examples rather than just weighting the existing ones more heavily. Class weights change the loss function; SMOTE changes the training data distribution. In practice, the results are often similar, but SMOTE is more widely discussed in the fraud detection literature.

**"Could this work in real time?"**
> Yes — and our demo shows exactly that. The LightGBM model takes milliseconds to score a transaction. In a production system, you'd wrap it in a streaming data pipeline — transactions come in, features get engineered on the fly, the model scores, and the result goes back to the payment processor before the customer even gets a response.

**"How do you know the model isn't just memorising the training data?"**
> The test set. We locked away 20% of the data before any training happened. The model never saw it. The metrics we reported — Recall 0.8399, F1 0.8807, AUC 0.9978 — are all from that locked test set. If the model were memorising, it would score lower there than on training data. We also used 5-fold cross-validation during tuning, which further guards against overfitting during hyperparameter search.

