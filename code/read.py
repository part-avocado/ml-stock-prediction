ticker = 'TSLA'
interations = 1000  # Please make sure that this is the same number as the filename
                    # eg. ltsm_TSLA_1000.weights.h5 would be:
                    # ticker = 'TSLA'
                    # iterations = 1000

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import yfinance as yf
from datetime import datetime, timedelta
import os
import time

# Step 1: Automatically load historical stock price data
def load_stock_data(ticker, months_prior=1):
    end_date = datetime.today()
    start_date = datetime(2020, 10, 9)  # Fixed: Using datetime object instead of arithmetic
    data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    data = data[['Close']]
    return data

# Step 2: Prepare the dataset for LSTM
def create_dataset(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Step 3: Generate future predictions
def predict_future(data, model, scaler, lookback, steps_ahead):
    last_sequence = data[-lookback:]
    predictions = []
    
    for _ in range(steps_ahead):
        prediction_input = np.reshape(last_sequence, (1, lookback, 1))
        next_price = model.predict(prediction_input, verbose=0)[0, 0]
        predictions.append(next_price)
        last_sequence = np.append(last_sequence[1:], next_price).reshape(-1, 1)
    
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Main execution
if __name__ == "__main__":
    # Load data
    data = load_stock_data(ticker, months_prior=12)
    prices = data['Close'].values.reshape(-1, 1)
    data['Date'] = data.index

    # Preprocess data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)

    # Create sequences
    lookback = 60
    X, y = create_dataset(scaled_prices, lookback)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Build model
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(units=50),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Reading weights
    weights_file = f'lstm_{ticker}_{interations}.weights.h5'  # Fixed filename format
    os.system('clear')
    try:
        path_to_file = f'/results/ltsm_{ticker}.weights.h5'
        print(f"Loading existing weights from {weights_file}")
        model.load_weights(weights_file)
    except FileNotFoundError:
        print(f"The weights for {ticker} could not be found. Please ensure that a file with the file name 'ltsm_{ticker}.weights.h5' exists.")
        exit(1)

    # Make predictions
    os.system('clear')
    print("Making predictions. This shouldn't take too long.")
    predict_start = time.time()
    predicted_prices = model.predict(X, verbose=0)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    predict_end = time.time()
    print(f"Finished predicting in {predict_end-predict_start} seconds.")

    # Generate future predictions
    steps_ahead = 126  # Approx. 6 months of trading days
    future_predictions = predict_future(scaled_prices, model, scaler, lookback, steps_ahead)

    # Save predictions to CSV
    prediction_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), 
                                   periods=steps_ahead, freq='B')
    future_df = pd.DataFrame({
        'Date': prediction_dates,
        'Predicted_Price': future_predictions.flatten()
    })
    
    output_file = f'{ticker}_{interations}_prediction.csv'
    future_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

    # Visualize results
    plt.figure(figsize=(14, 5))
    plt.plot(data['Date'][lookback:], scaler.inverse_transform(scaled_prices[lookback:]), 
             color='blue', label='Actual Prices')
    plt.plot(data['Date'][lookback:], predicted_prices, 
             color='red', label='Predicted Prices')
    plt.plot(future_df['Date'], future_df['Predicted_Price'], 
             color='green', label='6-Month Predictions')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Dates')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

