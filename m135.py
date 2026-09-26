"""
Titanic K-Means clustering demo (annotated).

What this script does:
- Loads a Titanic dataset from `titanic.xls` (expects file in CWD).
- Drops a couple of columns that are not useful for clustering.
- Converts non-numeric (categorical/text) columns into integer codes.
- Fills missing values with 0 so the clustering algorithm can run.
- Runs K-Means with 2 clusters and then measures how often the
  cluster assignment matches the `survived` column.

Notes:
- K-Means is unsupervised; comparing clusters to the `survived` label is
  only a rough way to see if the clustering aligns with the true labels.
- This is for illustration and learning, not a production classifier.
"""

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd


# --- Load dataset ---
# Reads the Titanic dataset; file must be present in the working directory.
df = pd.read_csv('titanic.xls')

# --- Drop unnecessary/unique columns ---
# `body` is noisy and `name` is effectively unique per passenger; both
# contribute little useful information for clustering here.
# The commented-out line below shows other columns you might drop.
# df.drop(['ticket', 'fare', 'cabin', 'embarked', 'home.dest'], 1, inplace=True)
df.drop(['body', 'name'], 1, inplace=True)

# The following line attempts to convert values to numeric where possible.
# Note: `convert_objects` is deprecated in modern pandas versions, but is
# present in some older example code; it is left here to preserve original
# behaviour from the tutorial this repo follows.
try:
    df.convert_objects(convert_numeric=True)
except Exception:
    # If the pandas version does not have convert_objects, ignore and continue.
    pass

# Replace missing values with 0 so the algorithm won't fail on NaNs.
df.fillna(value=0, inplace=True)


def handle_non_numeric_data(df):
    """Encode non-numeric columns to integer codes.

    This goes through each column and, if the column dtype is not numeric,
    creates a mapping from every unique text value to a small integer and
    replaces the column values with those integers.
    """
    columns = df.columns.values
    for column in columns:
        # Dictionary to hold mapping from text -> integer code.
        text_digit_vals = {}

        # Helper that will be used after building the mapping
        # to convert column values to the corresponding ints.
        def convert_to_int(value):
            return text_digit_vals[value]

        # Skip numeric columns
        if df[column].dtype == np.int64 or df[column].dtype == np.float64:
            continue

        # Build mapping of unique text values to integers
        column_contents = df[column].values.tolist()
        unique_elements = set(column_contents)

        x = 0
        for unique in unique_elements:
            if unique not in text_digit_vals:
                text_digit_vals[unique] = x
                x += 1

        # Replace the column values with their integer codes
        df[column] = list(map(convert_to_int, df[column]))

    return df


# Apply the encoding function to the dataframe
df = handle_non_numeric_data(df)

# At this point, `df` should contain only numeric values (or zeros for missing).
# Next we separate the features and the label.
X = np.array(df.drop(['survived'], 1).astype(float))  # feature matrix
Y = np.array(df['survived'])                         # labels (0 or 1)


# --- Clustering ---
# Use KMeans with 2 clusters because the target `survived` has two values.
clf = KMeans(n_clusters=2)
clf.fit(X)


# --- Evaluation (rough) ---
# For each sample, predict its cluster and check if the predicted cluster id
# equals the actual `survived` label. This gives a very rough "accuracy"
# indicating how well the unsupervised clusters align with the survival label.
correct = 0
for i in range(len(X)):
    # Prepare one sample in the shape expected by the model
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))

    response = clf.predict(predict_me)
    if response[0] == Y[i]:
        correct += 1

# Print the fraction of samples where cluster id matched the survival label
print(correct / len(X))
        
    