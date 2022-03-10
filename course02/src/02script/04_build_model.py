import pickle

import numpy as np
import pandas as pd
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 20, 10
from keras.models import Sequential
from keras.layers import LSTM, Dense

rcParams['figure.figsize'] = 20, 10
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('data/VNIndex.csv')
print(df.head())

df = df.rename(columns={"<DTYYYYMMDD>": "Date", "<Close>": "Close"})
df = df.astype({"Close": float})
df = df.astype({"Date": str})

df["Date"] = pd.to_datetime(df.Date, format="%Y%m%d")
print(df.dtypes)

# For our prediction project, we will just need “Date” and “Close” columns. Let’s get rid of the other columns then.
df = df[['Date', 'Close']]
print(df.head())

df = df.sort_index(ascending=True, axis=0)
df.reset_index(drop=True, inplace=True)

## copy the data

data = df.copy()

pickle.dump(data, open('/tmp/data.pkl', 'wb'))

scaler = MinMaxScaler(feature_range=(0, 1))
data.index = data.Date
data.drop('Date', axis=1, inplace=True)
final_data = data.values
train_data = final_data[0:2000, :]
valid_data = final_data[2000:, :]
scaler = MinMaxScaler(feature_range=(0, 1))

scaled_data = scaler.fit_transform(final_data)

pickle.dump(scaler, open('/tmp/scaler.pkl', 'wb'))

x_train_data, y_train_data = [], []
for i in range(60, len(train_data)):
    x_train_data.append(scaled_data[i - 60:i, 0])
    y_train_data.append(scaled_data[i, 0])

# LSTM Model
# In this step, we are defining the Long Short-Term Memory model.
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(np.shape(x_train_data)[1], 1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))
model_data = data[len(data) - len(valid_data) - 60:].values
model_data = model_data.reshape(-1, 1)
model_data = scaler.transform(model_data)

# Train and Test Data
# This step covers the preparation of the train data and the test data.
lstm_model.compile(loss='mean_squared_error', optimizer='adam')
lstm_model.fit(x=np.array(x_train_data), y=np.array(y_train_data), epochs=5, batch_size=1, verbose=2)

# save the model
pickle.dump(lstm_model, open('/tmp/lstm_model.pkl', 'wb'))

# history = model.fit(x=np.array(tr_X), y=np.array(tr_Y), epochs=3, validation_data=(np.array(va_X), np.array(va_Y)), batch_size=batch_size, steps_per_epoch=spe, validation_freq=5)
X_test = []
for i in range(60, model_data.shape[0]):
    X_test.append(model_data[i - 60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

pickle.dump(X_test, open('/tmp/xtest.pkl', 'wb'))
