"""
K-Nearest Neighbors (KNN) classification example.

What this script does:
- Loads the Wisconsin Breast Cancer dataset from `wdbc.data`.
- Replaces missing/unknown values (`?`) with a sentinel value so they do not
  disrupt the model.
- Removes the non-feature `id` column.
- Splits the data into training and testing sets.
- Trains a KNeighborsClassifier.
- Measures the model accuracy on the test set.
- Repeats this process 25 times and averages the accuracies to get a more
  stable estimate.

This is a basic supervised learning example for classification.
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, linear_model, svm, neighbors

# Run the model multiple times to get a more stable accuracy estimate.
accuracies = []
for i in range(25):
    # Load the dataset.
    df = pd.read_csv('wdbc.data')

    # Replace any missing values with a large negative number so the classifier
    # can treat them as outliers rather than valid data.
    df.replace('?', -99999, inplace=True)

    # Remove the ID column because it is just an identifier and not a feature.
    df.drop(['id'], axis=1, inplace=True)

    # Separate features (X) and target label (y).
    x = np.array(df.drop(['diagnosis'], axis=1))
    y = np.array(df['diagnosis'])

    # Split data into train/test sets.
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, test_size=0.2
    )

    # Create a KNN classifier with the default settings.
    clf = neighbors.KNeighborsClassifier()
    clf.fit(x_train, y_train)

    # Measure accuracy on the test set.
    accuracy = clf.score(x_test, y_test)
    # print(accuracy)

    # Store this run's accuracy.
    accuracies.append(accuracy)

# Print the average accuracy across all 25 runs.
print(sum(accuracies) / len(accuracies))