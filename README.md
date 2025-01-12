# ML Stock Predictor

A machine learning stock predictor that automatically downloads historical stock price data using the `yfinance` library, preprocesses the data by scaling the closing prices, and transforms the data into sequences suitable for training an LSTM (Long Short-Term Memory) model. The LSTM model is built and trained using TensorFlow's Keras API. Once trained, the model predicts future stock prices, scales the predictions back to the original price range, and saves the predicted prices for the next 6 months to a CSV file. Additionally, the script generates a plot showing the actual prices, predicted prices, and future predictions.

## How to Use
1. Ensure that you are in the `results` folder before running any code.
2. 