# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 11:16:54 2023

@author: star26
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load stock price data
# Replace 'your_stock_data.csv' with the actual file containing your stock price data
data = pd.read_excel(r'\\192.168.60.77\dataset_share\Stock_Price.xlsx')

data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data = data.sort_values(by='Date', ascending=True)

# Extract the closing prices
closing_prices = data['High'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
closing_prices_scaled = scaler.fit_transform(closing_prices)

# Prepare the data for LSTM
def create_sequences(data, seq_length, future_steps=1):
    sequences = []
    target = []
    for i in range(len(data) - seq_length - future_steps + 1):
        seq = data[i:i+seq_length]
        label = data[i+seq_length:i+seq_length+future_steps]
        sequences.append(seq)
        target.append(label)
    return np.array(sequences), np.array(target)

# Define sequence length (number of time steps to look back)
sequence_length = 10
future_steps = 18  # Number of future time steps to predict

# Create sequences and targets
X, y = create_sequences(closing_prices_scaled, sequence_length, future_steps)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(Dense(units=future_steps))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Make predictions for future time steps
future_sequence = closing_prices_scaled[-sequence_length:]
future_sequence = future_sequence.reshape((1, sequence_length, 1))
future_predictions = model.predict(future_sequence)

# Inverse transform the predictions to original scale
future_predictions = scaler.inverse_transform(future_predictions.reshape(-1, 1))

# Generate future dates
future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_steps, freq='B')

# Create DataFrame with future predictions and dates
future_df = pd.DataFrame({'Date': future_dates, 'Future_Predictions': future_predictions.flatten()})
future_df.set_index('Date', inplace=True)

# Display the DataFrame with future predictions
print(future_df)
