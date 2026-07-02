import numpy as np
import pandas as pd
import yfinance as yf
import math
from sklearn import preprocessing, model_selection, linear_model, svm
from sklearn.linear_model import LinearRegression
from matplotlib import style
import matplotlib.pyplot as plt
import pickle


style.use('ggplot')
df = yf.download("AAPL", start="2023-01-01", end="2026-06-20")

# Flatten MultiIndex columns when yfinance returns a ticker-level index
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Add useful computed features
df["HL_PCT"] = (df["High"] - df["Low"]) / df["Close"] * 100.0
df["PCT_change"] = (df["Close"] - df["Open"]) / df["Open"] * 100.0

# select important columns including computed percentages
cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'HL_PCT', 'PCT_change']
df = df.loc[:, [c for c in cols if c in df.columns]].dropna()
forecast_col = 'Close'

df.fillna(-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))
print(forecast_out)
df['label'] = df[forecast_col].shift(-forecast_out)
x = np.array(df.drop(['label'], axis=1))


x = preprocessing.scale(x)
x = x[:-forecast_out]
x_lately = x[-forecast_out:]
# x = x[:-forecast_out+1]
df.dropna(inplace=True)

x = np.array(df.drop(['label'], axis=1))
y = np.array(df['label'])

print(len(x), len(y))

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)
clf = LinearRegression(n_jobs=-1)
# clf = svm.SVR(kernel='poly')
clf.fit(x_train, y_train)
with open('linearregression.pickle', 'wb') as f:
    pickle.dump(clf, f)

pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(x_test, y_test)
print("accuracy: ", accuracy)

forecast_set = clf.predict(x_lately)
print(forecast_set, accuracy,forecast_out)

df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = pd.to_datetime(next_unix, unit='s')
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]
print(df.head())
df['Close'].plot()
df['Forecast'].plot()
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


# df['Forecast'].iloc[-forecast_out:] = forecast_set