"""
Logistic Regression basics with a simple example.

Logistic regression is used for binary classification problems, where the output
is usually 0 or 1.

Example:
- Predict whether a student passes an exam based on study hours.
- Predict whether a customer buys a product based on age and income.

The model learns a relationship between input features and a probability.
It then converts that probability to a class label using a threshold (usually 0.5).
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

# Example dataset:
# hours_studied -> whether the student passed (1 = passed, 0 = failed)
X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8],
    [9],
    [10]
])

y = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])

# Train the model.
model = LogisticRegression()
model.fit(X, y)

# Predict on some sample cases.
sample_cases = np.array([
    [2],
    [5],
    [8],
    [10]
])

predictions = model.predict(sample_cases)
probabilities = model.predict_proba(sample_cases)

print("Logistic Regression Example")
for i, value in enumerate(sample_cases):
    hours = value[0]
    prediction = predictions[i]
    prob = probabilities[i]
    prob_yes = round(prob[1], 3)
    prob_no = round(prob[0], 3)

    label = "Pass" if prediction == 1 else "Fail"
    print(f"Study hours: {hours} -> Prediction: {label} (P(pass)={prob_yes}, P(fail)={prob_no})")

X_2d = np.array([
    [1, 30],
    [2, 35],
    [3, 40],
    [4, 45],
    [5, 50],
    [6, 60],
    [7, 70],
    [8, 75]
])

y_2d = np.array([0, 0, 0, 1, 1, 1, 1, 1])

model_2d = LogisticRegression()
model_2d.fit(X_2d, y_2d)

# Sample cases to test.
new_cases = np.array([
    [2, 40],
    [5, 55],
    [7, 80]
])

print("Multiple-feature sample cases")
for case in new_cases:
    pred = model_2d.predict([case])[0]
    prob = model_2d.predict_proba([case])[0]
    print(f"Input: {case} -> Prediction: {'Yes' if pred == 1 else 'No'} | Probabilities: {np.round(prob, 3)}")

