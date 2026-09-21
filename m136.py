"""
Titanic survival clustering example.

This script loads the Titanic dataset, cleans it, converts categorical values into
numeric codes, applies K-Means clustering with two clusters, and then checks how
well the cluster assignment matches the actual survival label.

Important note:
    This is not a standard supervised classifier. The model is unsupervised, and
    we are evaluating whether the resulting clusters are aligned with the target
    variable `survived`.
"""

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd


def handle_non_numeric_data(df):
    """Convert any text/category columns to a numeric integer encoding."""
    columns = df.columns.values

    for column in columns:
        # Skip columns that are already numeric.
        if df[column].dtype == np.int64 or df[column].dtype == np.float64:
            continue

        # Map each unique text/category value to an integer code.
        text_digit_vals = {}

        def convert_to_int(value):
            return text_digit_vals[value]

        column_contents = df[column].values.tolist()
        unique_elements = set(column_contents)

        x = 0
        for unique in unique_elements:
            if unique not in text_digit_vals:
                text_digit_vals[unique] = x
                x += 1

        df[column] = list(map(convert_to_int, df[column]))

    return df


def main():
    # Load the dataset.
    # The Titanic dataset is expected to be available in the current working directory.
    df = pd.read_csv('titanic.xls')

    # Remove columns that are not useful for this clustering example.
    # The `body` and `name` columns are either irrelevant or too unique to help with
    # clustering patterns in this small demonstration.
    df.drop(columns=['body', 'name'], inplace=True)

    # Convert any object/string values to numeric codes before feeding data to KMeans.
    # Missing values are filled with 0 so the algorithm can operate without errors.
    df = handle_non_numeric_data(df)
    df.fillna(value=0, inplace=True)

    # Separate the feature matrix X from the target Y.
    # X contains all columns except the label "survived".
    X = np.array(df.drop(['survived'], axis=1).astype(float))
    Y = np.array(df['survived'])

    # KMeans with 2 clusters because the target variable has two possible classes:
    # 0 = did not survive, 1 = survived.
    clf = KMeans(n_clusters=2)
    clf.fit(X)

    # Evaluate how often the cluster assignment matches the actual survival label.
    correct = 0
    for i in range(len(X)):
        predict_me = np.array(X[i].astype(float)).reshape(-1, len(X[i]))
        cluster_response = clf.predict(predict_me)

        if cluster_response[0] == Y[i]:
            correct += 1

    accuracy = correct / len(X)
    print(f"KMeans accuracy against survival label: {accuracy:.4f}")


if __name__ == "__main__":
    main()

