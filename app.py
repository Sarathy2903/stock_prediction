import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf

# STOCK PREDICTOR CLASS

class StockPredictor:
    def __init__(self, symbol):
        self.symbol = symbol
        self.start_date = (datetime.now() - timedelta(days=120)).strftime('%Y-%m-%d')
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        self.scaler = MinMaxScaler()
        self.lstm_model = None
        self.arima_model = None
        self.data = None

    def fetch_data(self):
        stock = yf.Ticker(self.symbol)
        self.data = stock.history(start=self.start_date, end=self.end_date)
        return self.data[['Close', 'Volume']].values

    def prepare_lstm_data(self, data, lookback=60):
        scaled = self.scaler.fit_transform(data)
        X, y = [], []
        for i in range(lookback, len(scaled)):
            X.append(scaled[i-lookback:i])
            y.append(scaled[i, 0])
        return np.array(X), np.array(y)

    def build_lstm(self, lookback, features):
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(lookback, features)),
            Dropout(0.3),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def train_lstm(self, X, y):
        self.lstm_model = self.build_lstm(X.shape[1], X.shape[2])
        self.lstm_model.fit(X, y, epochs=20, batch_size=32, verbose=0)

    def train_arima(self, data):
        self.arima_model = ARIMA(data[:, 0], order=(1,1,1)).fit()

    def predict_lstm(self, X):
        preds = self.lstm_model.predict(X)
        dummy = np.zeros_like(preds)
        merged = np.concatenate([preds, dummy], axis=1)
        return self.scaler.inverse_transform(merged)[:, 0]

    def predict_arima(self, steps):
        return self.arima_model.forecast(steps=steps)

# STREAMLIT UI

st.set_page_config(page_title="Stock Price Predictor", layout="wide")

st.title("📈 Stock Price Prediction using LSTM + ARIMA")

st.sidebar.header("Input Parameters")

symbol = st.sidebar.text_input("Stock Symbol", value="AAPL")
future_days = st.sidebar.slider("Prediction Days", 3, 10, 5)

run = st.sidebar.button("Run Prediction")

if run:
    with st.spinner("Fetching data & training models..."):
        predictor = StockPredictor(symbol)
        data = predictor.fetch_data()

        X, y = predictor.prepare_lstm_data(data)
        split = int(len(X) * 0.8)

        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        predictor.train_lstm(X_train, y_train)
        predictor.train_arima(data)

        lstm_pred = predictor.predict_lstm(X_test)[:future_days]
        arima_pred = predictor.predict_arima(future_days)

        combined = 0.6 * lstm_pred + 0.4 * arima_pred

        dates = pd.date_range(start=datetime.now(), periods=future_days)

        df = pd.DataFrame({
            "Date": dates,
            "LSTM Prediction": lstm_pred,
            "ARIMA Prediction": arima_pred,
            "Final Prediction": combined
        })

    st.success("Prediction Completed ✅")

    # METRICS

    actual = predictor.data['Close'][split+60:split+60+future_days].values

    if len(actual) == len(combined):
        mse = mean_squared_error(actual, combined)
        mae = mean_absolute_error(actual, combined)
        accuracy = 100 - ((mae / np.mean(actual)) * 100)

        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE", f"{np.sqrt(mse):.2f}")
        col2.metric("MAE", f"{mae:.2f}")
        col3.metric("Accuracy", f"{accuracy:.2f}%")


    # PLOT
    st.subheader("📊 Stock Price Forecast")

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(predictor.data.index, predictor.data['Close'], label="Historical")
    ax.plot(dates, combined, '--', label="Predicted")
    ax.legend()
    ax.grid()

    st.pyplot(fig)


    # TABLE
    st.subheader("📋 Prediction Table")
    st.dataframe(df)
