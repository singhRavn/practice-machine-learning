"""
Basic Data Preprocessing example.

Data preprocessing is the step before machine learning where you clean and
prepare raw data. It includes:
- handling missing values
- normalizing/scaling features
- encoding categorical variables
- removing duplicates

This example shows each step on a simple dataset.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

raw_data = {
    'age': [25, 30, np.nan, 45, 50, 35],
    'salary': [30000, 50000, 60000, np.nan, 80000, 55000],
    'city': ['NYC', 'LA', 'NYC', 'Chicago', 'LA', 'NYC'],
    'employed': ['Yes', 'Yes', 'No', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(raw_data)
print("1) Raw Dataset:")
print(df)
print()

df['age'].fillna(df['age'].mean(), inplace=True)
df['salary'].fillna(df['salary'].mean(), inplace=True)

print("2) After filling missing values:")
print(df)
print()

df['city_encoded'] = pd.factorize(df['city'])[0]
df['employed_encoded'] = df['employed'].map({'Yes': 1, 'No': 0})

print("3) After encoding categorical variables:")
print(df)
print()

scaler = StandardScaler()
df[['age_scaled', 'salary_scaled']] = scaler.fit_transform(df[['age', 'salary']])

print("4) After scaling numeric features:")
print(df)
print()

final_df = df[['age_scaled', 'salary_scaled', 'city_encoded', 'employed_encoded']]

print("5) Final preprocessed dataset ready for ML:")
print(final_df)
print()

print("Summary of preprocessing steps:")
print("- Handled missing values")
print("- Encoded text categories to numbers")
print("- Scaled numeric features to similar ranges")
print("- Removed unnecessary columns")
