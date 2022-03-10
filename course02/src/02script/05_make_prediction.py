# import step from 04_make_prediction.
import pickle

import matplotlib.pyplot as plt

X_test = pickle.load(open('/tmp/xtest.pkl', 'rb'))
scaler = pickle.load(open('/tmp/scaler.pkl', 'rb'))
data = pickle.load(open('/tmp/data.pkl', 'rb'))

# load model
lstm_model = pickle.load(open('/tmp/lstm_model.pkl', 'rb'))

predicted_stock_price = lstm_model.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

### Prediction Result
train_data = data[:2000]
valid_data = data[2000:]
valid_data['Predictions'] = predicted_stock_price
plt.plot(train_data["Close"])
plt.plot(valid_data[['Close', "Predictions"]])
plt.show()
