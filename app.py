# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import ta  # Using ta instead of talib
import gdown  # For downloading from Google Drive
import yfinance as yf
from datetime import datetime, timedelta
import warnings
from sklearn.exceptions import InconsistentVersionWarning
import logging
import time
import os  # Added import for os

# Ensure the 'models' directory exists
os.makedirs('models', exist_ok=True)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# NIFTY 50 companies and their ticker symbols
nifty_50_companies = {
    'Adani Ports and Special Economic Zone Ltd': 'ADANIPORTS.NS',
    'Asian Paints Ltd': 'ASIANPAINT.NS',
    'Axis Bank Ltd': 'AXISBANK.NS',
    'Bajaj Auto Ltd': 'BAJAJ-AUTO.NS',
    'Bajaj Finance Ltd': 'BAJFINANCE.NS',
    'Bajaj Finserv Ltd': 'BAJAJFINSV.NS',
    'Bharti Airtel Ltd': 'BHARTIARTL.NS',
    'Britannia Industries Ltd': 'BRITANNIA.NS',
    'Cipla Ltd': 'CIPLA.NS',
    'Coal India Ltd': 'COALINDIA.NS',
    "Divi's Laboratories Ltd": 'DIVISLAB.NS',
    "Dr. Reddy's Laboratories Ltd": 'DRREDDY.NS',
    'Eicher Motors Ltd': 'EICHERMOT.NS',
    'Grasim Industries Ltd': 'GRASIM.NS',
    'HCL Technologies Ltd': 'HCLTECH.NS',
    'HDFC Bank Ltd': 'HDFCBANK.NS',
    'HDFC Life Insurance Company Ltd': 'HDFCLIFE.NS',
    'Hero MotoCorp Ltd': 'HEROMOTOCO.NS',
    'Hindalco Industries Ltd': 'HINDALCO.NS',
    'Hindustan Unilever Ltd': 'HINDUNILVR.NS',
    'Housing Development Finance Corporation Ltd': 'HDFC.NS',
    'ICICI Bank Ltd': 'ICICIBANK.NS',
    'ITC Ltd': 'ITC.NS',
    'IndusInd Bank Ltd': 'INDUSINDBK.NS',
    'Infosys Ltd': 'INFY.NS',
    'JSW Steel Ltd': 'JSWSTEEL.NS',
    'Kotak Mahindra Bank Ltd': 'KOTAKBANK.NS',
    'Larsen & Toubro Ltd': 'LT.NS',
    'Mahindra & Mahindra Ltd': 'M&M.NS',
    'Maruti Suzuki India Ltd': 'MARUTI.NS',
    'Nestle India Ltd': 'NESTLEIND.NS',
    'Oil & Natural Gas Corporation Ltd': 'ONGC.NS',
    'Power Grid Corporation of India Ltd': 'POWERGRID.NS',
    'Reliance Industries Ltd': 'RELIANCE.NS',
    'State Bank of India': 'SBIN.NS',
    'Sun Pharmaceutical Industries Ltd': 'SUNPHARMA.NS',
    'Tata Consultancy Services Ltd': 'TCS.NS',
    'Tata Consumer Products Ltd': 'TATACONSUM.NS',
    'Tata Motors Ltd': 'TATAMOTORS.NS',
    'Tata Steel Ltd': 'TATASTEEL.NS',
    'Tech Mahindra Ltd': 'TECHM.NS',
    'Titan Company Ltd': 'TITAN.NS',
    'UltraTech Cement Ltd': 'ULTRACEMCO.NS',
    'UPL Ltd': 'UPL.NS',
    'Wipro Ltd': 'WIPRO.NS',
}

# Define the list of features
feature_columns = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'MACD', 'MACD_Signal',
    'MACD_Hist', 'RSI', 'Upper_BB', 'Middle_BB', 'Lower_BB', 'SlowK', 'SlowD'
]

# Google Drive file IDs for the models
MODEL_FILES = {
    "random_forest": "1xnR8oLEBWwFVAQGxo-N60unE4_qYwyau",
    "xgboost": "1xr5Jd8d1s1nGPnusoc4V7qIGYCIpMhQX",
    "voting_classifier": "YOUR_VOTING_CLASSIFIER_FILE_ID",
    "scaler": "1TivUPsc6uw84f9wpMYzH3X805Gkotn6V",
    "label_encoder": "10UNoD9wl5UjKKGA_ciEg_W3FfjMvPP4S"
}

# Helper function to download from Google Drive
def download_from_drive(file_id, output):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output, quiet=False)

# Caching the model loading to improve performance
def load_model(model_name):
    file_name = f"models/{model_name}.pkl"
    try:
        # Check if the file exists locally
        model = joblib.load(file_name)
    except FileNotFoundError:
        # Download from Google Drive if not found
        download_from_drive(MODEL_FILES[model_name], file_name)
        model = joblib.load(file_name)
    return model

# Load scaler and label encoder
def load_scaler():
    try:
        scaler = joblib.load("models/scaler.pkl")
    except FileNotFoundError:
        download_from_drive(MODEL_FILES["scaler"], "models/scaler.pkl")
        scaler = joblib.load("models/scaler.pkl")
    return scaler

def load_label_encoder():
    try:
        label_encoder = joblib.load("models/label_encoder.pkl")
    except FileNotFoundError:
        download_from_drive(MODEL_FILES["label_encoder"], "models/label_encoder.pkl")
        label_encoder = joblib.load("models/label_encoder.pkl")
    return label_encoder

# Cache data fetching from yfinance
@st.cache_data
def fetch_data(ticker_symbol, start_date, end_date):
    data = yf.download(ticker_symbol, start=start_date, end=end_date)
    return data

# Cache technical indicator calculations using ta
@st.cache_data
def calculate_indicators(data):
    df = data.copy()

    # Ensure there are enough data points
    if len(df) < 30:
        return None

    # MACD
    macd = ta.trend.MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()

    # RSI
    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()

    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['Upper_BB'] = bollinger.bollinger_hband()
    df['Middle_BB'] = bollinger.bollinger_mavg()
    df['Lower_BB'] = bollinger.bollinger_lband()

    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(
        high=df['High'], low=df['Low'], close=df['Close'],
        window=14, smooth_window=3
    )
    df['SlowK'] = stoch.stoch()
    df['SlowD'] = stoch.stoch_signal()

    # Get the latest indicators
    latest_indicators = {
        'MACD': df['MACD'].iloc[-1],
        'MACD_Signal': df['MACD_Signal'].iloc[-1],
        'MACD_Hist': df['MACD_Hist'].iloc[-1],
        'RSI': df['RSI'].iloc[-1],
        'Upper_BB': df['Upper_BB'].iloc[-1],
        'Middle_BB': df['Middle_BB'].iloc[-1],
        'Lower_BB': df['Lower_BB'].iloc[-1],
        'SlowK': df['SlowK'].iloc[-1],
        'SlowD': df['SlowD'].iloc[-1]
    }

    return latest_indicators

# Function to get the selected model
def get_model(model_choice):
    model_map = {
        'Random Forest': "random_forest",
        'XGBoost': "xgboost",
        'Ensemble Voting': "voting_classifier"
    }
    return load_model(model_map[model_choice])

def main():
    st.title('Stock Buy/Hold/Sell Recommendation')

    # Model selection
    st.sidebar.header('Model Selection')
    model_choice = st.sidebar.selectbox(
        'Select the model to use for prediction:',
        ('Random Forest', 'XGBoost', 'Ensemble Voting')
    )

    # Company selection
    st.sidebar.header('Company Selection')
    company_name = st.sidebar.selectbox(
        'Select a NIFTY 50 company:',
        list(nifty_50_companies.keys())
    )
    ticker_symbol = nifty_50_companies[company_name]

    # Date selection
    st.sidebar.header('Date Selection')
    end_date = st.sidebar.date_input('Select End Date', datetime.today())
    start_date = end_date - timedelta(days=60)

    # Load models and encoders
    with st.spinner('Loading models and encoders...'):
        selected_model = get_model(model_choice)
        scaler = load_scaler()
        label_encoder = load_label_encoder()

    # Fetch data using yfinance
    with st.spinner('Fetching data...'):
        data = fetch_data(ticker_symbol, start_date, end_date)

    if data.empty:
        st.error("No data found for the selected date range.")
        return

    # Use the most recent data for Open, High, Low, Close, Volume
    latest_data = data.iloc[-1]
    open_price = latest_data['Open']
    high_price = latest_data['High']
    low_price = latest_data['Low']
    close_price = latest_data['Close']
    volume = latest_data['Volume']

    # Calculate technical indicators
    with st.spinner('Calculating technical indicators...'):
        indicators = calculate_indicators(data)

    if indicators is None:
        st.error("Not enough data to calculate technical indicators.")
        return

    # Prepare input data
    input_data = pd.DataFrame({
        'Open': [open_price],
        'High': [high_price],
        'Low': [low_price],
        'Close': [close_price],
        'Volume': [volume],
        'MACD': [indicators['MACD']],
        'MACD_Signal': [indicators['MACD_Signal']],
        'MACD_Hist': [indicators['MACD_Hist']],
        'RSI': [indicators['RSI']],
        'Upper_BB': [indicators['Upper_BB']],
        'Middle_BB': [indicators['Middle_BB']],
        'Lower_BB': [indicators['Lower_BB']],
        'SlowK': [indicators['SlowK']],
        'SlowD': [indicators['SlowD']]
    })

    # Handle missing values if any
    input_data.fillna(method='bfill', inplace=True)
    input_data.fillna(method='ffill', inplace=True)

    # Feature scaling
    input_scaled = scaler.transform(input_data)

    # Predict with selected model
    with st.spinner('Making predictions...'):
        prediction_proba = selected_model.predict_proba(input_scaled)
        prediction = selected_model.predict(input_scaled)
        prediction_label = label_encoder.inverse_transform(prediction)[0]

    # Display prediction
    st.header('Prediction')
    st.write(f"The model recommends to **{prediction_label}** the stock.")

    # Display prediction probabilities
    st.header('Prediction Probabilities')
    prob_df = pd.DataFrame(prediction_proba, columns=label_encoder.classes_)
    st.write(prob_df.T.rename(columns={0: 'Probability'}))

    # Optionally, display recent stock data
    if st.checkbox('Show recent stock data'):
        st.subheader('Recent Stock Data')
        st.dataframe(data.tail(10))

    # Optionally, display technical indicators
    if st.checkbox('Show technical indicators'):
        st.subheader('Technical Indicators')
        st.write(pd.DataFrame(indicators, index=[0]))

if __name__ == '__main__':
    main()
