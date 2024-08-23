import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
import matplotlib.pyplot as plt
import csv

#file_url = 'https://raw.githubusercontent.com/LearnPythonWithRune/MachineLearningWithPython/main/files/aapl.csv'
file = open('aapl.csv')
data = pd.read_csv(file, parse_dates=True, index_col=0)

# Create a train and test set
data_train = data.loc['2000':'2020', 'Adj Close'].to_numpy()
data_test = data.loc['2021', 'Adj Close'].to_numpy()

# Use the MinMaxScaler to scale the data
scaler = MinMaxScaler()
data_train = scaler.fit_transform(data_train.reshape(-1, 1))
data_test = scaler.transform(data_test.reshape(-1, 1))

# To divide data into x and y set
def data_preparation(data):
    x = []
    y = []
    
    for i in range(40, len(data)):
        x.append(data[i-40:i, 0])
        y.append(data[i])
        
    x = np.array(x)
    y = np.array(y)
    
    x = x.reshape(x.shape[0], x.shape[1], 1)
    
    return x, y

x_train, y_train = data_preparation(data_train)
x_test, y_test = data_preparation(data_test)

# Create the model
model = Sequential()
model.add(LSTM(units=45, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=45, return_sequences=True))
model.add(LSTM(units=45))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Predict with the model
y_pred = model.predict(x_test)

# Unscale it
y_unscaled = scaler.inverse_transform(y_pred)

# See the prediction accuracy
fig, ax = plt.subplots()
y_real = data.loc['2021', 'Adj Close'].to_numpy()
ax.plot(y_real[40:])
ax.plot(y_unscaled)
plt.show()