# Regression Learning

This repository contains a small collection of Python scripts for learning and experimenting with regression, classification, and basic machine learning concepts.

## What’s in this repo

- `DataTest.py` - downloads historical Apple stock data using `yfinance`, creates simple technical indicators, trains a linear regression model, and plots a forecast.
- `BestFitSlop.py` - demonstrates simple linear regression from scratch using synthetic data and computes $R^2$.
- `Euclidean_Distance.py` - implements a custom K-nearest neighbors classifier from scratch for the Wisconsin Breast Cancer dataset.
- `K_nearest_neighbour.py` - uses scikit-learn’s `KNeighborsClassifier` on the same breast cancer dataset.
- `SVM_from_scratch.py` - shows a basic support vector machine implementation from scratch.
- `wdbc.names` - metadata for the Wisconsin Breast Cancer dataset used by the KNN and SVM examples.

## Requirements

Make sure you have Python 3.9+ installed.

## Setup

From the project folder, create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

## Install dependencies

```bash
pip install --upgrade pip
pip install numpy pandas matplotlib scikit-learn yfinance
```

## Dataset note

The KNN and SVM example scripts expect a file named `wdbc.data` in the project directory. That file is not included in this repository, but you can download it from the UCI Machine Learning Repository (Wisconsin Breast Cancer Dataset) and place it beside `wdbc.names`.

## Run the scripts

```bash
python DataTest.py
python BestFitSlop.py
python Euclidean_Distance.py
python K_nearest_neighbour.py
python SVM_from_scratch.py
```

## Project goals

These scripts are intended as educational examples for:

- regression and forecasting
- feature engineering
- linear regression
- K-nearest neighbors
- support vector machines
