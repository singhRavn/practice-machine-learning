"""
Logistic Regression: one step advanced from the basic example.

In the previous version, we trained logistic regression on a small dataset and
looked at predictions.

Now we go one step further:
- split the data into training and testing sets
- train the model on training data
- evaluate it on unseen test data
- make sample predictions to see how the model behaves

This is a more realistic ML workflow.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -------------------------------------------------------------------
# 1) Build a simple binary classification dataset
# -------------------------------------------------------------------
# Example: whether a student passes based on study hours
X = np.array([
    [1], [2], [3], [4], [5],
    [6], [7], [8], [9], [10],
    [11], [12], [13], [14], [15]
])

y = np.array([0, 0, 0, 0, 1,
              0, 1, 1, 1, 1,
              1, 1, 1, 1, 1])

# -------------------------------------------------------------------
# 2) Split into train and test sets
# -------------------------------------------------------------------
# 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------------------------------------
# 3) Train logistic regression model
# -------------------------------------------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -------------------------------------------------------------------
# 4) Evaluate the model on unseen test data
# -------------------------------------------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Logistic Regression (Advanced Example)")
print("-----------------------------------")
print(f"Train size: {len(X_train)}")
print(f"Test size: {len(X_test)}")
print(f"Accuracy: {accuracy:.2f}")

# -------------------------------------------------------------------
# 5) Sample predictions for new cases
# -------------------------------------------------------------------
sample_cases = np.array([
    [2],
    [5],
    [8],
    [12]
])

predicted_labels = model.predict(sample_cases)
probabilities = model.predict_proba(sample_cases)

for i, case in enumerate(sample_cases):
    hours = case[0]
    label = "Pass" if predicted_labels[i] == 1 else "Fail"
    p_pass = round(probabilities[i][1], 3)
    p_fail = round(probabilities[i][0], 3)
    print(f"Study hours = {hours} -> Prediction: {label} | P(pass)={p_pass}, P(fail)={p_fail}")

# -------------------------------------------------------------------
# 6) Another example with two input features
# -------------------------------------------------------------------
# Here, the model uses age and income to decide whether a customer buys.
X_2d = np.array([
    [20, 2000],
    [25, 2500],
    [30, 3000],
    [35, 4500],
    [40, 5000],
    [45, 6000],
    [50, 7000],
    [55, 8000]
])

y_2d = np.array([0, 0, 0, 1, 1, 1, 1, 1])

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X_2d, y_2d, test_size=0.25, random_state=42
)

model_2d = LogisticRegression()
model_2d.fit(X2_train, y2_train)

new_cases = np.array([
    [28, 2600],
    [41, 5200],
    [60, 8500]
])

print("\nSecond example: customer purchase prediction")
for case in new_cases:
    prediction = model_2d.predict([case])[0]
    probability = model_2d.predict_proba([case])[0]
    label = "Buy" if prediction == 1 else "No Buy"
    print(f"Input = {case} -> {label} | Probabilities = {np.round(probability, 3)}")

# -------------------------------------------------------------------
# 7) What this shows
# -------------------------------------------------------------------
# Logistic regression gives a probability for each class and then chooses
# the most likely class using a decision threshold (usually 0.5). It works
# well for binary classification problems.

