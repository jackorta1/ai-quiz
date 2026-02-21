# COMP237 FINAL EXAM PREDICTION
## Machine Learning & Logistic Regression Focus

**Strategy:** This exam is heavily focused on Module 5 (ML). Based on exam.md patterns + professor tendencies, here are the most likely questions organized by priority.

---
---

# SECTION A: DEFINITIONS & SHORT ANSWER (Expect 6-10 questions)

---

## A1. Logistic Regression [GUARANTEED]

**Q: What is Logistic Regression used for?**
> Used for **binary classification** problems. It predicts the **probability** that an observation belongs to a particular class (0 or 1) using the sigmoid function.

**Q: What is the sigmoid function and why is it used?**
> sigma(z) = 1 / (1 + e^(-z))
>
> It transforms any real number into a value **between 0 and 1**, converting the linear output into a **probability**.

**Q: What is the difference between Linear Regression and Logistic Regression?**

| Linear Regression | Logistic Regression |
|---|---|
| Predicts **continuous** values | Predicts **probabilities** (0 to 1) |
| Output: any real number | Output: between 0 and 1 |
| Uses **MSE** (L2) loss | Uses **Log Loss** (Cross-Entropy) |
| Regression problem | Classification problem |

**Q: Why can't we use MSE for Logistic Regression?**
> Because the sigmoid function is **non-linear**. Using MSE would create a **non-convex** optimization problem with many local minima. We use **Log Loss** instead, which is convex.

---

## A2. Loss Functions [VERY LIKELY]

**Q: Write the Log Loss (Cross-Entropy) formula.**
> Loss = -[y * log(p) + (1 - y) * log(1 - p)]
>
> Where: y = actual label (0 or 1), p = predicted probability

**Q: What happens to Log Loss when the prediction is correct vs wrong?**
> - If y=1 and p is close to 1 --> Loss is close to 0 (good)
> - If y=1 and p is close to 0 --> Loss is very large (bad, penalized heavily)
> - If y=0 and p is close to 0 --> Loss is close to 0 (good)
> - If y=0 and p is close to 1 --> Loss is very large (bad, penalized heavily)

**Q: Name the 3 types of loss functions from the course.**
> - L1 = |y - y_hat| (absolute loss)
> - L2 = (y - y_hat)^2 (squared loss)
> - L0/1 = 0 if y = y_hat, else 1 (0-1 loss for classification)

---

## A3. Types of Learning [LIKELY]

**Q: What are the 3 main types of machine learning?**
> 1. **Supervised Learning** -- learns from labeled input-output pairs (teacher provides labels)
> 2. **Unsupervised Learning** -- no labels; finds patterns/clusters on its own
> 3. **Reinforcement Learning** -- learns from rewards and punishments through experience

**Q: What is the difference between classification and regression?**
> - **Classification**: output is **categorical/discrete** (e.g., spam/not spam, cat/dog)
> - **Regression**: output is **continuous** (e.g., temperature = 33.5, price = $450)

**Q: Give an example of each type of learning.**
> - Supervised: Email spam detection (labeled emails)
> - Unsupervised: Customer segmentation (grouping without labels)
> - Reinforcement: Self-driving car learning to navigate

---

## A4. Overfitting & Bias-Variance [GUARANTEED]

**Q: What is overfitting?**
> When a model performs **very well on training data** but **poorly on new/unseen data**. The model has learned the noise in the training data rather than the underlying pattern.

**Q: What is underfitting?**
> When the model is **too simple** to capture the underlying pattern. It performs poorly on **both** training and test data.

**Q: Explain the Bias-Variance tradeoff.**
> - **High Bias** = model is too simple, **underfits** (misses patterns)
> - **High Variance** = model is too complex, **overfits** (fits noise)
> - Goal: find the **balance** -- a model complex enough to capture patterns but simple enough to generalize

**Q: How can we detect overfitting?**
> When training accuracy is **much higher** than test accuracy, the model is likely overfitting.

---

## A5. Train/Test Split & K-Fold [GUARANTEED]

**Q: Why do we split data into training and testing sets?**
> To evaluate how the model performs on **unseen data** and to **prevent/detect overfitting**.

**Q: What is a typical train-test split ratio?**
> 80% training, 20% testing (or 70/30). The model **never sees** the test data during training.

**Q: Explain K-Fold Cross Validation.**
> 1. Dataset is divided into **K equal parts** (folds)
> 2. Each fold is used **once** as the test set
> 3. The remaining **K-1 folds** are used for training
> 4. This is repeated K times, and results are **averaged**
> 5. Provides a more **reliable** performance estimate than a single split

**Q: What is the advantage of K-Fold over a single train-test split?**
> Every data point gets used for both training and testing. This gives a **more reliable estimate** and reduces the impact of how the data happens to be split.

---

# SECTION B: CONFUSION MATRIX & METRICS (Expect 2-3 questions, ALMOST GUARANTEED)

---

## B1. Definitions

**Q: Define TP, TN, FP, FN.**
> - **TP (True Positive):** Predicted positive, actually positive (correct)
> - **TN (True Negative):** Predicted negative, actually negative (correct)
> - **FP (False Positive):** Predicted positive, actually negative (Type I Error)
> - **FN (False Negative):** Predicted negative, actually positive (Type II Error)

**Confusion Matrix Layout:**
```
                 Predicted
              Positive  Negative
Actual  Pos [   TP    |   FN   ]
        Neg [   FP    |   TN   ]
```

---

## B2. Formulas [MUST MEMORIZE]

**Accuracy** = (TP + TN) / (TP + TN + FP + FN)
> "Of ALL predictions, how many were correct?"

**Precision** = TP / (TP + FP)
> "Of all POSITIVE predictions, how many were actually positive?"

**Recall (Sensitivity)** = TP / (TP + FN)
> "Of all ACTUAL positives, how many did we catch?"

**F1 Score** = 2 * (Precision * Recall) / (Precision + Recall)
> Harmonic mean of Precision and Recall (balances both)

---

## B3. Calculation Practice

### Practice 1 (from exam.md style):
Given: TP=30, TN=50, FP=10, FN=10 (Total=100)

- **Accuracy** = (30+50)/100 = **80/100 = 0.80 = 80%**
- **Precision** = 30/(30+10) = **30/40 = 0.75 = 75%**
- **Recall** = 30/(30+10) = **30/40 = 0.75 = 75%**

### Practice 2 (new):
Given: TP=45, TN=30, FP=15, FN=10 (Total=100)

- **Accuracy** = (45+30)/100 = **75/100 = 0.75 = 75%**
- **Precision** = 45/(45+15) = **45/60 = 0.75 = 75%**
- **Recall** = 45/(45+10) = **45/55 = 0.818 = 81.8%**

### Practice 3 (tricky - imbalanced):
Given: TP=5, TN=900, FP=10, FN=85 (Total=1000)

- **Accuracy** = (5+900)/1000 = **905/1000 = 90.5%** (looks good but...)
- **Precision** = 5/(5+10) = **5/15 = 0.33 = 33.3%** (terrible!)
- **Recall** = 5/(5+85) = **5/90 = 0.056 = 5.6%** (terrible!)
- **Lesson: High accuracy does NOT always mean a good model!** (imbalanced data)

### Practice 4 (another new one):
Given: TP=60, TN=20, FP=5, FN=15 (Total=100)

- **Accuracy** = (60+20)/100 = **80/100 = 0.80 = 80%**
- **Precision** = 60/(60+5) = **60/65 = 0.923 = 92.3%**
- **Recall** = 60/(60+15) = **60/75 = 0.80 = 80%**

---

## B4. Precision vs Recall [VERY LIKELY]

**Q: When is Precision more important?**
> When **False Positives are costly**.
> - Spam filter: don't want important emails marked as spam
> - Criminal conviction: don't want to convict an innocent person

**Q: When is Recall more important?**
> When **False Negatives are costly**.
> - Cancer detection: don't want to miss a sick patient
> - Airport security: don't want to miss a threat
> - Fraud detection: don't want to miss fraud

**Q: Why might accuracy alone be misleading?**
> In **imbalanced datasets** (e.g., 99% negative, 1% positive), a model that predicts "negative" every time gets 99% accuracy but catches 0% of positive cases. Precision and Recall give a better picture.

---

# SECTION C: ROC & AUC [VERY LIKELY]

---

**Q: What does the ROC curve show?**
> The tradeoff between **True Positive Rate (Recall/Sensitivity)** on the Y-axis and **False Positive Rate** on the X-axis at different classification thresholds.

**Q: What is the ideal ROC curve?**
> A curve that hugs the **top-left corner** (high TPR, low FPR). This means the model correctly identifies positives without false alarms.

**Q: What does AUC mean?**
> **Area Under the ROC Curve**. It measures overall model performance in a single number.
> - AUC = **1.0** --> Perfect model
> - AUC = **0.5** --> Random guessing (diagonal line)
> - AUC **< 0.5** --> Worse than random
> - AUC = **0.8-0.9** --> Good model

**Q: If the ROC curve is close to the diagonal, what does it mean?**
> The model is **no better than random guessing** -- it's a poor model.

**Q: What are True Positive Rate and False Positive Rate?**
> - TPR (Sensitivity/Recall) = TP / (TP + FN) -- "How many actual positives did we catch?"
> - FPR = FP / (FP + TN) -- "How many actual negatives did we falsely flag?"

---

# SECTION D: DATA PREPROCESSING [VERY LIKELY]

---

## D1. Feature Scaling

**Q: Why is feature scaling important?**
> To ensure all features **contribute equally** to the model. Without scaling, features with large values (e.g., salary in thousands) would **dominate** features with small values (e.g., age). Especially important for distance-based models and Logistic Regression.

**Q: What is the difference between Normalization and Standardization?**

| Normalization (Min-Max) | Standardization (Z-score) |
|---|---|
| x' = (x - min) / (max - min) | z = (x - mu) / sigma |
| Scales to range **[0, 1]** | Centers around **mean=0, std=1** |
| Sensitive to outliers | Less sensitive to outliers |
| Good when data has known bounds | Good for normally distributed data |

**Q: Should scaling be done BEFORE or AFTER train-test split?** [TRICK QUESTION - FAVORITE]
> **AFTER splitting!**
> 1. Split the data first
> 2. Fit the scaler on the **training data ONLY**
> 3. Transform both training and test data using the training scaler
>
> **Why?** If you scale before splitting, the test data "leaks" information into training (data leakage), giving unrealistically good results.

---

## D2. Missing Values & Feature Engineering

**Q: How do we handle missing values?**
> 1. **Remove** rows/columns with missing values
> 2. **Replace** with mean, median, or mode
> 3. Use **predictive models** to fill in values

**Q: What percentage of time is typically spent on feature engineering?**
> **70-80%** of the total ML pipeline time.

**Q: What is feature engineering?**
> Preparing the feature space: data transformation, scaling, handling missing values, correlation analysis, and feature selection.

---

# SECTION E: STATISTICS & LINEAR REGRESSION [LIKELY]

---

## E1. Correlation

**Q: What is correlation?**
> A measure of the **relationship between two variables**. Range: **-1 to +1**.
> - **+1** = perfect positive correlation (both increase together)
> - **-1** = perfect negative correlation (one increases, other decreases)
> - **0** = no correlation
> - Above **0.5** or below **-0.5** = notable correlation

---

## E2. Hypothesis Testing

**Q: What is the null hypothesis (H0)?**
> The default assumption that **no effect/relationship** exists. E.g., H0: w1 = 0 (no relationship between x and y).

**Q: What is the p-value?**
> The probability of observing the data (or more extreme) **if H0 is true**. If p-value < 0.05, we **reject H0**.

**Q: Difference between T-test and Z-test?**

| T-test | Z-test |
|---|---|
| Population variance **unknown** | Population variance **known** |
| Sample size **< 30** | Sample size **> 30** |
| Student-t distribution | Normal distribution |

**Q: What is the Chi-square test used for?**
> 1. Test **independence** between input and output variables
> 2. Check if observed data is from an **unbiased source**
> 3. Formula: chi^2 = sum of (O - E)^2 / E

---

## E3. Linear Regression

**Q: What is the linear regression equation?**
> y = w1*x + w0 (univariate)
> - w0 = intercept (y-axis)
> - w1 = slope

**Q: What is R-squared?**
> R^2 = 1 - (SSE/SST)
> Measures how well the model explains variability.
> - R^2 = 1 --> perfect fit
> - R^2 = 0 --> model explains nothing

**Q: What is SST, SSR, SSE?**
> - **SST** (Total) = sum of (y_actual - y_mean)^2 -- total variability
> - **SSR** (Regression) = variability explained by the model
> - **SSE** (Error) = residual/unexplained variability
> - SST = SSR + SSE

---

# SECTION F: PYTHON CODING [IF PRACTICAL EXAM]

---

## F1. Full Pipeline (memorize this order!)

```python
# 1. IMPORTS
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score, roc_curve)
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

# 2. LOAD DATA
data = load_breast_cancer()
X = data.data
y = data.target

# 3. SPLIT DATA (BEFORE scaling!)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. SCALE DATA (fit on train ONLY)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit + transform
X_test_scaled = scaler.transform(X_test)          # transform only!

# 5. TRAIN MODEL
model = LogisticRegression(max_iter=10000)
model.fit(X_train_scaled, y_train)

# 6. PREDICT
y_pred = model.predict(X_test_scaled)

# 7. CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
print(cm)  # [[TN, FP], [FN, TP]]

# 8. METRICS
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))

# 9. ROC CURVE
y_prob = model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()

# 10. K-FOLD CROSS VALIDATION
scores = cross_val_score(model, X, y, cv=10)
print("Average Accuracy:", scores.mean())
```

**CRITICAL ORDER:** Split --> Scale (fit on train only) --> Train --> Predict --> Evaluate

---

## F2. Key Coding Traps

| Mistake | Correct Way |
|---|---|
| Scale BEFORE splitting | Split first, THEN scale |
| fit_transform on test data | Only `transform()` on test data |
| Forget `max_iter` for LogReg | Use `max_iter=10000` |
| Use `predict` for ROC | Use `predict_proba()[:, 1]` for ROC |
| Confuse confusion matrix layout | sklearn: [[TN, FP], [FN, TP]] |

---

# SECTION G: TRICK QUESTIONS & TRUE/FALSE [EXPECT 3-5]

---

**"High accuracy always means a good model."**
> **FALSE.** With imbalanced data, a useless model can have high accuracy (e.g., always predict majority class).

**"ROC curve below the diagonal means a good model."**
> **FALSE.** Below diagonal = **worse than random**. Good model = above diagonal, toward top-left.

**"K-Fold Cross Validation eliminates overfitting."**
> **FALSE.** It **reduces bias** in evaluation and gives a better estimate of performance, but does NOT eliminate overfitting. The model can still overfit.

**"Feature scaling should be done before the train-test split."**
> **FALSE.** Scale **after** splitting. Fit scaler on training data only to avoid data leakage.

**"Logistic Regression is a regression algorithm."**
> **FALSE (tricky!).** Despite the name, Logistic Regression is used for **classification**, not regression.

**"A model with 100% training accuracy is ideal."**
> **FALSE.** It likely means the model is **overfitting** -- memorizing the training data instead of learning general patterns.

**"Precision and Recall are always equally important."**
> **FALSE.** It depends on the problem. Cancer detection = recall matters more. Spam filter = precision matters more.

**"Increasing K in K-Fold always improves the model."**
> **FALSE.** Very high K (e.g., Leave-One-Out) is computationally expensive and may have high variance in estimates.

**"Normalization and Standardization are the same thing."**
> **FALSE.** Normalization scales to [0,1]; Standardization centers around mean=0, std=1. They serve different purposes.

**"If correlation between two variables is 0, they are independent."**
> **FALSE (tricky!).** Zero correlation means no **linear** relationship. There could still be a **non-linear** relationship.

---

# SECTION H: MY EXTRA PREDICTIONS (Beyond exam.md)

---

## H1. Questions the Professor Likely Adds

**Q: What is the decision boundary in Logistic Regression?**
> The threshold (usually 0.5) where the model switches prediction from class 0 to class 1. If sigmoid output >= 0.5, predict class 1; otherwise predict class 0.

**Q: What is data leakage and why is it dangerous?**
> When information from outside the training set is used to create the model. Example: scaling before splitting lets test data statistics influence training. This gives **overly optimistic** results that won't hold on truly new data.

**Q: What is the F1 Score and when should you use it?**
> F1 = 2 * (Precision * Recall) / (Precision + Recall)
> Use it when you need to **balance** Precision and Recall, especially with imbalanced classes.

**Q: Compare Logistic Regression with Linear Regression in terms of loss functions.**
> - Linear Regression: L2 loss (MSE) = sum of (y - y_hat)^2 --> convex, easy to optimize
> - Logistic Regression: Log Loss = -[y*log(p) + (1-y)*log(1-p)] --> also convex with sigmoid

**Q: What is the hypothesis space?**
> The set of all possible functions h that a learning algorithm can choose from. The goal is to find the h in H that best approximates the true function f.

**Q: What is a convex function and why does it matter?**
> A function where any line segment between two points lies above the curve. It has a **single global minimum** -- no local minima to get stuck in. This is why we use Log Loss for Logistic Regression instead of MSE.

**Q: In the confusion matrix, what are Type I and Type II errors?**
> - **Type I Error** = False Positive (FP) -- rejecting H0 when it's true
> - **Type II Error** = False Negative (FN) -- failing to reject H0 when it's false

**Q: What is the purpose of the `random_state` parameter in train_test_split?**
> It sets the random seed so the split is **reproducible** -- the same split every time you run the code.

---

## H2. Scenario-Based Questions (Professor Favorite)

**Q: You're building a model to detect cancer. Which metric matters most: Accuracy, Precision, or Recall? Why?**
> **Recall.** Missing a cancer patient (False Negative) is far worse than a false alarm (False Positive). We want to catch every positive case.

**Q: Your model has 95% accuracy but only 20% recall. Is it a good model?**
> **No.** It means the model misses 80% of actual positive cases. The high accuracy is likely due to class imbalance (most samples are negative).

**Q: You trained a model that gets 99% accuracy on training data but 60% on test data. What's happening?**
> **Overfitting.** The model memorized the training data but can't generalize to new data. Solutions: use simpler model, get more training data, use regularization, or use K-Fold CV.

**Q: Why do we use `fit_transform()` on training data but only `transform()` on test data?**
> `fit_transform()` **learns** the parameters (mean, std) from training data and applies them. `transform()` uses those **same parameters** on test data. This prevents data leakage.

---

# QUICK FORMULA CHEAT SHEET

```
Sigmoid:         sigma(z) = 1 / (1 + e^(-z))
Log Loss:        L = -[y*log(p) + (1-y)*log(1-p)]
Accuracy:        (TP + TN) / (TP + TN + FP + FN)
Precision:       TP / (TP + FP)
Recall:          TP / (TP + FN)
F1 Score:        2 * (P * R) / (P + R)
Normalization:   (x - min) / (max - min)
Standardization: (x - mu) / sigma
Linear Reg:      y = w1*x + w0
R-squared:       1 - (SSE / SST)
Chi-square:      sum of (O - E)^2 / E
Z-test:          Z = (sample_mean - pop_mean) / (sigma / sqrt(n))
```

---

# EXAM DAY PRIORITY ORDER

If short on time, study in this order:

1. **Confusion Matrix calculations** (TP/TN/FP/FN + Accuracy/Precision/Recall)
2. **Precision vs Recall** (when each matters + examples)
3. **ROC & AUC** (what they mean, what AUC values mean)
4. **Train/Test Split vs K-Fold** (why, how, order)
5. **Feature Scaling** (normalize vs standardize, AFTER split)
6. **Overfitting** (definition, detection, bias-variance)
7. **Logistic Regression** (sigmoid, Log Loss, vs Linear Regression)
8. **Trick Questions** (Section G above -- read ALL of them)
9. **Python code pipeline** (memorize the correct order)
10. **Types of learning** (supervised/unsupervised/reinforcement)
