# app.py
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import talib
from keras.models import load_model
import yfinance as yf
from datetime import datetime, timedelta

# Optional: Check library versions for compatibility
import sklearn
import tensorflow as tf

st.sidebar.title("Configuration")

@st.cache_resource
def load_lstm_model(path):
    return load_model(path)

@st.cache_resource
def load_scaler(path):
    return joblib.load(path)

@st.cache_resource
def load_label_encoder(path):
    return joblib.load(path)

# Load models and encoders
try:
    model = load_lstm_model('models_deep_learning_/lstm_model.h5')
    scaler = load_scaler('models_deep_learning_/scaler.pkl')
    label_encoder = load_label_encoder('models_deep_learning_/label_encoder.pkl')  # Ensure this file exists
except FileNotFoundError as e:
    st.error(f"Model file not found: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Define the list of features
feature_columns = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'MACD', 'MACD_Signal',
    'MACD_Hist', 'RSI', 'Upper_BB', 'Middle_BB', 'Lower_BB', 'SlowK', 'SlowD'
]

sequence_length = 30  # Same as used during training

# Streamlit app
st.title('📈 Stock Buy/Hold/Sell Recommendation (LSTM Model)')

# User inputs
st.header('🔍 Enter Stock Symbol and Date')

# Stock symbol input
stock_symbol = st.text_input('Stock Symbol (e.g., AAPL for Apple Inc.)', value='AAPL').upper()

# Exchange selection
exchange = st.selectbox('📈 Select Exchange', options=['NSE', 'BSE', 'NYSE', 'NASDAQ', 'Other'])

# Map exchange to suffix
exchange_suffix = {
    'NSE': '.NS',
    'BSE': '.BO',
    'NYSE': '',
    'NASDAQ': '',
    'Other': ''
}

# Append the suffix
suffix = exchange_suffix.get(exchange, '')
stock_symbol_full = stock_symbol + suffix

# Date input
selected_date = st.date_input('📅 Select Date', value=datetime.today())

# Prediction button
if st.button('🔮 Predict'):
    with st.spinner('Fetching data and making prediction...'):
        try:
            # Calculate the start and end dates for fetching data
            end_date = selected_date
            start_date = end_date - timedelta(days=150)  # Increased buffer for indicator calculations

            # Fetch historical data using yfinance
            df = yf.download(stock_symbol_full, start=start_date, end=end_date)

            if df.empty:
                st.error(f"No data found for {stock_symbol_full}. Please check the symbol and try again.")
            else:
                # Reset index to make 'Date' a column
                df.reset_index(inplace=True)

                # Ensure the data is sorted by date
                df.sort_values('Date', inplace=True)

                # Calculate technical indicators
                df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(
                    df['Close'], fastperiod=12, slowperiod=26, signalperiod=9
                )
                df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
                df['Upper_BB'], df['Middle_BB'], df['Lower_BB'] = talib.BBANDS(
                    df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
                )
                df['SlowK'], df['SlowD'] = talib.STOCH(
                    df['High'], df['Low'], df['Close'],
                    fastk_period=14, slowk_period=3, slowk_matype=0,
                    slowd_period=3, slowd_matype=0
                )

                # Drop rows with NaN values resulting from indicator calculations
                df.dropna(inplace=True)

                # Use the last 'sequence_length' days
                if len(df) < sequence_length:
                    st.error(f"Not enough data available for {stock_symbol_full} to make a prediction.")
                else:
                    df_input = df.tail(sequence_length).copy()

                    # Ensure we have all required features
                    if not set(feature_columns).issubset(df_input.columns):
                        st.error("Missing required technical indicators. Please check the data.")
                    else:
                        # Prepare input data
                        X_input = df_input[feature_columns].values

                        # Scale input data
                        X_input_scaled = scaler.transform(X_input)

                        # Reshape to match model input
                        X_input_scaled = X_input_scaled.reshape(1, sequence_length, len(feature_columns))

                        # Make prediction
                        prediction_probs = model.predict(X_input_scaled)
                        prediction_class = np.argmax(prediction_probs, axis=1)
                        prediction_label = label_encoder.inverse_transform(prediction_class)[0]

                        # Display prediction
                        st.header('📊 Prediction')
                        st.write(f"The model recommends to **{prediction_label}** the stock.")

                        # Display prediction probabilities
                        st.header('🔢 Prediction Probabilities')
                        prob_df = pd.DataFrame(prediction_probs, columns=label_encoder.classes_)
                        prob_df = prob_df.T
                        prob_df.columns = ['Probability']
                        prob_df['Probability'] = prob_df['Probability'].round(4)
                        st.write(prob_df)

        except Exception as e:
            st.error(f"Error in prediction: {e}")