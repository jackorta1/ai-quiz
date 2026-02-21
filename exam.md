PART 1: Logistic Regression
❓ Q1: What is Logistic Regression used for?
Answer:
Logistic Regression is used for classification problems.
It predicts the probability that an observation belongs to a particular class (usually binary: 0 or 1).
❓ Q2: What is the sigmoid function?
Answer:
The sigmoid function transforms any real value into a number between 0 and 1.
σ
(
z
)
=
1
1
+
e
−
z
σ(z)= 
1+e 
−z
 
1
​	
 
It converts the linear equation output into a probability.
❓ Q3: What is the difference between Linear and Logistic Regression?
Answer:
Linear Regression	Logistic Regression
Predicts continuous values	Predicts probabilities
Output can be any real number	Output between 0 and 1
Uses MSE loss	Uses Log Loss
🔵 PART 2: Loss Function
❓ Q4: Why can't we use MSE in Logistic Regression?
Answer:
Because Logistic Regression uses a sigmoid function, which is non-linear.
Using MSE would result in a non-convex optimization problem.
Therefore, we use Log Loss (Cross-Entropy Loss).
❓ Q5: Write the formula for Log Loss.
Answer:
L
o
s
s
=
−
[
y
log
⁡
(
p
)
+
(
1
−
y
)
log
⁡
(
1
−
p
)
]
Loss=−[ylog(p)+(1−y)log(1−p)]
Where:
y = actual label (0 or 1)
p = predicted probability
🔵 PART 3: Classification Model Evaluation
❓ Q6: Define TP, TN, FP, FN.
Answer:
TP: True Positive (correct positive prediction)
TN: True Negative (correct negative prediction)
FP: False Positive (incorrect positive prediction)
FN: False Negative (incorrect negative prediction)
❓ Q7: Write formulas for Accuracy, Precision, and Recall.
Answer:
Accuracy:
A
c
c
u
r
a
c
y
=
T
P
+
T
N
T
P
+
T
N
+
F
P
+
F
N
Accuracy= 
TP+TN+FP+FN
TP+TN
​	
 
Precision:
P
r
e
c
i
s
i
o
n
=
T
P
T
P
+
F
P
Precision= 
TP+FP
TP
​	
 
Recall:
R
e
c
a
l
l
=
T
P
T
P
+
F
N
Recall= 
TP+FN
TP
​	
 
❓ Q8: When is Precision more important than Recall?
Answer:
Precision is more important when False Positives are costly.
Example:
Spam detection — we don’t want to mark important emails as spam.
❓ Q9: When is Recall more important than Precision?
Answer:
Recall is more important when False Negatives are costly.
Example:
Disease detection — we don’t want to miss a sick patient.
❓ Q10: What is an ROC curve?
Answer:
ROC curve is a graph that shows the tradeoff between:
True Positive Rate (Sensitivity)
False Positive Rate
The closer the curve is to the top-left corner, the better the model.
❓ Q11: What does AUC represent?
Answer:
AUC (Area Under Curve) measures the overall performance of the model.
AUC = 1 → Perfect model
AUC = 0.5 → Random guessing
AUC < 0.5 → Worse than random
🔵 PART 4: Train/Test Split & K-Fold
❓ Q12: Why do we split data into training and testing sets?
Answer:
To evaluate how the model performs on unseen data and prevent overfitting.
❓ Q13: What is Overfitting?
Answer:
Overfitting occurs when the model performs very well on training data but poorly on new data.
❓ Q14: Explain K-Fold Cross Validation.
Answer:
In K-Fold Cross Validation:
The dataset is divided into K equal parts.
Each part is used once as validation data.
The remaining K-1 parts are used for training.
The results are averaged.
This provides a more reliable performance estimate.
🔵 PART 5: Data Pre-processing
❓ Q15: Why is feature scaling important?
Answer:
Feature scaling ensures that all features contribute equally to the model and prevents features with large values from dominating.
❓ Q16: What is the difference between Normalization and Standardization?
Answer:
Normalization:
x
′
=
x
−
m
i
n
m
a
x
−
m
i
n
x 
′
 = 
max−min
x−min
​	
 
Scales data between 0 and 1.
Standardization:
z
=
x
−
μ
σ
z= 
σ
x−μ
​	
 
Centers data around mean 0 with standard deviation 1.
❓ Q17: How do we handle missing values?
Answer:
Remove rows with missing values
Replace with mean/median
Use predictive models
🔵 PART 6: Data Visualization
❓ Q18: Why is data visualization important?
Answer:
It helps to:
Understand data distribution
Detect outliers
Identify relationships between variables
❓ Q19: Name three common plots used in data visualization.
Answer:
Histogram
Scatter Plot
Boxplot
🔥 BONUS: Calculation Question (Professor Style)
❓ Q20:
Suppose:
TP = 40
TN = 50
FP = 10
FN = 20
Calculate:
Accuracy
Precision
Recall
✅ Answer:
Total = 40 + 50 + 10 + 20 = 120
Accuracy:
(
40
+
50
)
/
120
=
90
/
120
=
0.75
=
75
(40+50)/120=90/120=0.75=75
Precision:
40
/
(
40
+
10
)
=
40
/
50
=
0.80
=
80
40/(40+10)=40/50=0.80=80
Recall:
40
/
(
40
+
20
)
=
40
/
60
=
0.67
=
67
40/(40+20)=40/60=0.67=67


Question 1 — Train/Test Split
❓ Task:
Load a dataset and split it into training and testing sets.
🧠 Skills tested:
train_test_split
preparing ML data
✅ Python Solution:
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.shape)
print(X_test.shape)
✅ Question 2 — Train Logistic Regression Model
❓ Task:
Train a Logistic Regression model using training data.
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
✅ Question 3 — Make Predictions
❓ Task:
Predict class labels using test data.
y_pred = model.predict(X_test)
✅ Question 4 — Confusion Matrix
❓ Task:
Display confusion matrix.
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)
Output example:
[[TN FP]
 [FN TP]]
✅ Question 5 — Accuracy, Precision, Recall
from sklearn.metrics import accuracy_score, precision_score, recall_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
✅ Question 6 — ROC Curve
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

y_prob = model.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()
✅ Question 7 — K-Fold Cross Validation
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=10)

print(scores)
print("Average Accuracy:", scores.mean())
✅ Question 8 — Data Scaling (Pre-processing)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
✅ Question 9 — Visualization (Data Understanding)
import matplotlib.pyplot as plt

plt.hist(X[:,0])
plt.title("Feature Distribution")
plt.show()
🎯 PROFESSOR TIPS (VERY IMPORTANT)
Many students lose marks because:
❌ They scale BEFORE splitting data.
Correct order:
Split data
→ Fit scaler on TRAIN only
→ Transform test data



✅ Q1: What is Logistic Regression used for?
Expected answer:
Used for binary classification.
Outputs probability between 0 and 1 using sigmoid function.
✅ Q2: Write the sigmoid function formula.
σ
(
z
)
=
1
1
+
e
−
z
σ(z)= 
1+e 
−z
 
1
​	
 
They may ask:
Why is sigmoid used?
Answer:
To convert linear output into probability.
✅ Q3: What is overfitting?
Expected answer:
Model performs very well on training data but poorly on new unseen data.
✅ Q4: Why do we split data into train and test sets?
Answer:
To evaluate performance on unseen data.
To prevent overfitting.
✅ Q5: Explain K-Fold Cross Validation.
Answer:
Dataset is divided into K parts.
Each part is used once for testing.
The remaining K-1 parts are used for training.
Results are averaged.
🔵 PART 2 — CONFUSION MATRIX QUESTIONS (VERY LIKELY)
You will almost 100% get one like this.
✅ Q6: Define TP, TN, FP, FN.
Be ready to explain clearly.
✅ Q7: Given this confusion matrix:
TP = 30
TN = 50
FP = 10
FN = 10
Calculate:
Accuracy
Precision
Recall
You must know formulas.
Expected formulas:
Accuracy:
T
P
+
T
N
T
o
t
a
l
Total
TP+TN
​	
 
Precision:
T
P
T
P
+
F
P
TP+FP
TP
​	
 
Recall:
T
P
T
P
+
F
N
TP+FN
TP
​	
 
🔵 PART 3 — ROC & AUC QUESTIONS (VERY COMMON)
✅ Q8: What does ROC curve represent?
Answer:
Trade-off between True Positive Rate and False Positive Rate.
✅ Q9: What does AUC = 0.5 mean?
Answer:
Model performs like random guessing.
✅ Q10: If ROC curve is close to diagonal line, what does it mean?
Answer:
Poor model.
🔵 PART 4 — DATA PREPROCESSING QUESTIONS
These are very common.
✅ Q11: Why is feature scaling important?
Answer:
To prevent features with large values from dominating.
Important for models like Logistic Regression.
✅ Q12: Difference between normalization and standardization?
Normalization:
(
x
−
m
i
n
)
/
(
m
a
x
−
m
i
n
)
(x−min)/(max−min)
Standardization:
(
x
−
μ
)
/
σ
(x−μ)/σ
✅ Q13: Should we scale before or after train-test split?
Correct answer:
After splitting.
Fit scaler only on training data.
This is a very tricky question professors love.
🔵 PART 5 — SHORT ANSWER CONCEPT QUESTIONS
✅ Q14: When is precision more important than recall?
Answer:
When false positives are costly (e.g., spam filter).
✅ Q15: When is recall more important than precision?
Answer:
When false negatives are costly (e.g., cancer detection).
✅ Q16: Why don’t we use MSE for Logistic Regression?
Answer:
Because sigmoid is non-linear.
MSE causes non-convex optimization.
🔵 PART 6 — CODING QUESTIONS (If practical exam)
You may be asked to:
Use train_test_split
Train logistic regression
Print confusion matrix
Calculate accuracy
Plot ROC curve
Use cross_val_score
🔥 MOST IMPORTANT TOPICS (HIGH PRIORITY)
If you are short on time, focus on:
1️⃣ Confusion Matrix
2️⃣ Precision vs Recall
3️⃣ ROC & AUC
4️⃣ Train/Test vs K-Fold
5️⃣ Feature Scaling
6️⃣ Overfitting
These are almost guaranteed.
⚠️ PROFESSOR TRICK QUESTIONS
Be careful with:
👉 “High accuracy always means good model.”
Answer: False.
👉 “ROC below diagonal means good model.”
Answer: False.
👉 “K-Fold eliminates overfitting.”
Answer: False (it reduces bias, but does not eliminate).
👉 “Scaling should be done before splitting.”
Answer: False.
🎯 If I Predict Your Exam
Most likely structure:
Section A — Definitions
Section B — Confusion Matrix calculation
Section C — Explain ROC
Section D — Train/Test or K-Fold
Section E — Preprocessing