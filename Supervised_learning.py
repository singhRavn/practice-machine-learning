"""
Basic Supervised Learning example.

Supervised learning means the model learns from labeled examples.
Each sample has input features and a known correct output.

In this example, we use a simple dataset where:
- input = study hours
- output = pass/fail (1 or 0)

We train a logistic regression model on this data and then test it on
new sample cases.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ------------------------------------------------------------
# 1) Create a simple labeled dataset
# ------------------------------------------------------------
# Study hours and whether the student passed the exam
X = np.array([
    [1], [2], [3], [4], [5],
    [6], [7], [8], [9], [10],
    [11], [12], [13], [14], [15]
])

y = np.array([
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1
])

# ------------------------------------------------------------
# 2) Split into training and testing data
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------
# 3) Train the supervised learning model
# ------------------------------------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# ------------------------------------------------------------
# 4) Evaluate the model
# ------------------------------------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Supervised Learning Example")
print("--------------------------")
print(f"Accuracy: {accuracy:.2f}")

# ------------------------------------------------------------
# 5) Sample cases for prediction
# ------------------------------------------------------------
sample_cases = np.array([
    [2],
    [5],
    [8],
    [12]
])

predictions = model.predict(sample_cases)
probabilities = model.predict_proba(sample_cases)

for i, case in enumerate(sample_cases):
    study_hours = case[0]
    result = "Pass" if predictions[i] == 1 else "Fail"
    p_pass = round(probabilities[i][1], 3)
    p_fail = round(probabilities[i][0], 3)
    print(f"Study hours = {study_hours} -> {result} | P(pass)={p_pass}, P(fail)={p_fail}")

# ------------------------------------------------------------
# 6) Why this is supervised learning
# ------------------------------------------------------------
# Because the model learns from labeled data.
# Each example has the correct answer already known.
# The algorithm learns a mapping from inputs to outputs.
# Then it uses that learned mapping to predict unseen data.
