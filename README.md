# Regression Forecasting & Prediction

This project is a simple Python-based regression and forecasting project that uses stock market data from Yahoo Finance and performs basic feature engineering.

## Features
- Downloads historical stock data with `yfinance`
- Creates basic technical indicators such as:
  - `HL_PCT`
  - `PCT_change`
- Prepares a dataset for regression/forecasting experiments

## Project Structure
- `DataTest.py` - main script for downloading stock data and preparing features

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
pip install pandas yfinance
```

## Run the project

```bash
python DataTest.py
```
