import sys
import time
import datetime
import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

# For reading stock data from yahoo
from pandas_datareader.data import DataReader

# For time stamps
from datetime import datetime

# This method predicts stock prices based on 50 day moving day average and trend.
def analyze_stock():
  sns.set_style('whitegrid')
  plt.style.use("fivethirtyeight")

  # For reading stock data from yahoo
  from pandas_datareader.data import DataReader

  # For time stamps
  from datetime import datetime

  stockName = input("enter stock name: ")

  end = datetime.now()
  start = datetime(end.year - 3, end.month, end.day)
  ds = DataReader(stockName, 'yahoo', start, end)

  csv = stockName + '.csv'
  print(csv)

  ds.to_csv(csv)

  ds = pd.read_csv(csv)
  ds.set_index(pd.DatetimeIndex(ds['Date']), inplace=True)
  ds.ta.ema(close='Adj Close', length=50, append=True)
  ds.ta.adx(close='Adj Close', length=10, append=True)
  ds['delta'] = ds['Adj Close'] - ds['EMA_50']

  ds['Adj Close 1'] = ds['Adj Close']
  ds[['EMA_50', 'ADX_10', 'delta', 'Adj Close']] = ds[['EMA_50', 'ADX_10', 'delta', 'Adj Close']].shift(1)
  ds = ds.iloc[51:]

  #print(ds.head(30))
  #ds[['Adj Close', 'EMA_10']].plot()
  #plt.show()
  #print(qs)
  ds = ds.drop(columns=['High', 'Low', 'Close' , 'Volume', 'Open', 'Date'])
  #print(ds.tail(20))


  # Split data into testing and training sets
  X_train, X_test, y_train, y_test = train_test_split(ds[['EMA_50', 'ADX_10', 'delta']], ds[['Adj Close 1']], test_size=.2)
  # Test set

  from sklearn.linear_model import LinearRegression
  # Create Regression Model
  model = LinearRegression()
  # Train the model
  model.fit(X_train, y_train)

  # predict using the model

  y_pred = model.predict(X_train)


  #print (y_pred)

  from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


  #print("Training error:", mean_absolute_error(y_train, y_pred))

  y_pred = model.predict(X_test)
  #print("Test error:", mean_absolute_error(y_test, y_pred))



  model.fit(X_train, y_train)


  # Printout relevant metrics
  # print("Model Coefficients:", model.coef_)
  # print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
  # print("Coefficient of Determination:", r2_score(y_test, y_pred))

  ds2 = ds.tail(30)
  #print(ds2.tail(5))
  x_test2 = ds2[['EMA_50', 'ADX_10', 'delta']]
  y_actual = ds2[['Adj Close']]
  y_actualShift = ds2[['Adj Close 1']]
  y_pred2 = model.predict(x_test2)    
  final = y_pred2


  final = y_pred2 - y_actual
  final['real'] = ds['Adj Close'].tail(30)
  final['pred'] = y_pred2
  final['Shift'] = y_actualShift
  final["gain"] = final['Shift'] - final['real']
  final['pred gain'] = final['pred'] - final['real']
  final.loc[final['pred gain'] < 0.1, "gain"] = 0
  total = final['gain'].sum()
  print(final)
  print(final.describe())
  print(total)


# This method predicts stock prices based on the previous 40 day closing prices.
def analyze_stock_lstm():
  stockName = input("enter stock name: ")

  end = datetime.now()
  start = datetime(end.year - 3, end.month, end.day)
  ds = DataReader(stockName, 'yahoo', start, end)

  csv = stockName + '.csv'

  ds.to_csv(csv)

  ds = pd.read_csv(csv)
  ds = ds.drop(columns=['Close', 'Open', 'High', 'Low', 'Volume'])

  # create training and test sets
  # Will use last 40 values to predict 41st day price
  # Last 30 days of the input are used for testing
  ts = ds.iloc[:len(ds)-30, 1:2].values
  test_set = ds.iloc[len(ds)-70:, 1:2].values

  from sklearn.preprocessing import MinMaxScaler
  sc = MinMaxScaler(feature_range = (0, 1))
  ts = sc.fit_transform(ts)
  test_set = sc.fit_transform(test_set)
 
  x_train = []
  y_train = []
  x_test = []
  for i in range(40, len(ds)-30):
    x_train.append(ts[i-40:i, 0])
    y_train.append(ts[i, 0])
  x_train, y_train = np.array(x_train), np.array(y_train) 
  x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
  for i in range(len(test_set) - 30, len(test_set)):
    x_test.append(test_set[i-40:i, 0])
  x_test = np.array(x_test) 
  x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

  from keras.models import Sequential
  from keras.layers import Dense
  from keras.layers import LSTM
  from keras.layers import Dropout

  #creates the LSTM model

  regressor = Sequential()
  regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
  regressor.add(Dropout(0.2))
  regressor.add(LSTM(units = 50, return_sequences = True))
  regressor.add(Dropout(0.2))
  regressor.add(LSTM(units = 50, return_sequences = True))
  regressor.add(Dropout(0.2))
  regressor.add(LSTM(units = 50))
  regressor.add(Dropout(0.2))
  regressor.add(Dense(units = 1))
  regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

  regressor.fit(x_train, y_train, epochs = 100, batch_size = 32)

  predicted_stock_price = regressor.predict(x_test)
  predicted_stock_price = sc.inverse_transform(predicted_stock_price)

  #Testing the model using the last 30 days of a stock.

  ds['yDay'] = ds['Adj Close']
  ds['yDay'] = ds['yDay'].shift(1)

  ds2 = ds.tail(30)
  ds2['predict'] = predicted_stock_price
  ds2['buy'] = ds2['predict'] - ds2['yDay']
  ds2['profit'] = ds2['Adj Close'] - ds2['yDay']

  ds2.loc[ds2['buy'] <= 0.2, 'profit'] = 0
  print(ds2)

  print(ds2['profit'].sum())

  print("Mean Absolute Error:", mean_absolute_error(ds2['Adj Close'], ds2['predict']))
  print("Coefficient of Determination:", r2_score(ds2['Adj Close'], ds2['predict']))
  
  
  
# for i in range(1, len(sys.argv)):
#   tickers = sys.argv[i]
#   print(tickers)
analyze_stock()
#analyze_stock_lstm()
