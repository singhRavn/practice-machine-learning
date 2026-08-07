# Regression Learning

This repository contains a small collection of Python scripts for learning and experimenting with regression, classification, and basic machine learning techniques.

## What’s in this repo

- `DataTest.py` - downloads historical Apple stock data using `yfinance`, creates simple technical indicators, trains a linear regression model, and plots a forecast.
- `BestFitSlop.py` - demonstrates simple linear regression from scratch using synthetic data and computes the coefficient of determination ($R^2$).
- `Euclidean_Distance.py` - implements a custom K-nearest neighbors classifier from scratch on the Wisconsin Breast Cancer dataset.
- `K_nearest_neighbour.py` - demonstrates scikit-learn’s `KNeighborsClassifier` on the same breast cancer data.
- `SVM_from_scratch.py` - shows a basic support vector machine implementation from scratch with a small sample dataset.
- `wdbc.names` - metadata for the Wisconsin Breast Cancer dataset used by the classification examples.

## Requirements

- Python 3.9 or newer
- `pip` available for installing dependencies

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

For the KNN and SVM examples, the repository expects a file named `wdbc.data` in the project directory. That file is not included here. Download it from the UCI Machine Learning Repository (Wisconsin Breast Cancer Dataset) and place it next to `wdbc.names`.

## Run the scripts

```bash
python DataTest.py
python BestFitSlop.py
python Euclidean_Distance.py
python K_nearest_neighbour.py
python SVM_from_scratch.py
```

## Project goals

These scripts are intended to illustrate:

- regression and forecasting workflows
- basic feature engineering
- linear regression from scratch and with scikit-learn
- K-nearest neighbors classification
- support vector machine concepts
