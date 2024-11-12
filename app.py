# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import talib
import warnings
from sklearn.exceptions import InconsistentVersionWarning
import yfinance as yf
from datetime import datetime, timedelta

# Suppress specific warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# Caching the model loading to improve performance
@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_resource
def load_scaler(path):
    return joblib.load(path)

@st.cache_resource
def load_label_encoder(path):
    return joblib.load(path)

# Load models
final_rf_model = load_model('models/random_forest_final.pkl')
final_xgb_model = load_model('models/xgboost_final.pkl')
final_voting_clf = load_model('models/voting_classifier_final.pkl')
scaler = load_scaler('models/scaler_final.pkl')
label_encoder = load_label_encoder('models/label_encoder.pkl')

# Define the list of features
feature_columns = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'MACD', 'MACD_Signal',
    'MACD_Hist', 'RSI', 'Upper_BB', 'Middle_BB', 'Lower_BB', 'SlowK', 'SlowD'
]

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

# Streamlit app
st.title('Stock Buy/Hold/Sell Recommendation')

# Model selection
st.sidebar.header('Model Selection')
model_choice = st.sidebar.selectbox(
    'Select the model to use for prediction:',
    ('Random Forest', 'XGBoost', 'Ensemble Voting')
)

if model_choice == 'Random Forest':
    selected_model = final_rf_model
elif model_choice == 'XGBoost':
    selected_model = final_xgb_model
else:
    selected_model = final_voting_clf

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
start_date = end_date - timedelta(days=60)  # Fetch data for the past 60 days

# Fetch data using yfinance
try:
    data = yf.download(ticker_symbol, start=start_date, end=end_date)
    if data.empty:
        st.error("No data found for the selected date range.")
    else:
        # Use the most recent data for Open, High, Low, Close, Volume
        latest_data = data.iloc[-1]
        open_price = latest_data['Open']
        high_price = latest_data['High']
        low_price = latest_data['Low']
        close_price = latest_data['Close']
        volume = latest_data['Volume']

        # Get the closing prices for the last 30 days
        closing_prices = data['Close'][-30:].values

        if len(closing_prices) < 30:
            st.error("Not enough data to calculate technical indicators.")
        else:
            # Calculate technical indicators
            price_series = closing_prices

            # Calculate MACD
            macd, macd_signal, macd_hist = talib.MACD(
                price_series, fastperiod=12, slowperiod=26, signalperiod=9
            )
            macd = macd[-1]
            macd_signal = macd_signal[-1]
            macd_hist = macd_hist[-1]

            # Calculate RSI
            rsi = talib.RSI(price_series, timeperiod=14)
            rsi = rsi[-1]

            # Calculate Bollinger Bands
            upper_bb, middle_bb, lower_bb = talib.BBANDS(
                price_series, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
            )
            upper_bb = upper_bb[-1]
            middle_bb = middle_bb[-1]
            lower_bb = lower_bb[-1]

            # Get high and low prices for Stochastic Oscillator
            high_prices = data['High'][-30:].values
            low_prices = data['Low'][-30:].values

            # Calculate Stochastic Oscillator
            slowk, slowd = talib.STOCH(
                high=high_prices,
                low=low_prices,
                close=price_series,
                fastk_period=14,
                slowk_period=3,
                slowk_matype=0,
                slowd_period=3,
                slowd_matype=0
            )
            slowk = slowk[-1]
            slowd = slowd[-1]

            # Prepare input data
            input_data = pd.DataFrame({
                'Open': [open_price],
                'High': [high_price],
                'Low': [low_price],
                'Close': [close_price],
                'Volume': [volume],
                'MACD': [macd],
                'MACD_Signal': [macd_signal],
                'MACD_Hist': [macd_hist],
                'RSI': [rsi],
                'Upper_BB': [upper_bb],
                'Middle_BB': [middle_bb],
                'Lower_BB': [lower_bb],
                'SlowK': [slowk],
                'SlowD': [slowd]
            })

            # Handle missing values if any
            input_data.fillna(method='bfill', inplace=True)
            input_data.fillna(method='ffill', inplace=True)

            # Feature scaling
            input_scaled = scaler.transform(input_data)

            # Predict with selected model
            prediction_proba = selected_model.predict_proba(input_scaled)
            prediction = selected_model.predict(input_scaled)
            prediction_label = label_encoder.inverse_transform(prediction)[0]

            # Display prediction
            st.header('Prediction')
            st.write(f"The model recommends to **{prediction_label}** the stock.")

            # Display prediction probabilities
            st.header('Prediction Probabilities')
            prob_df = pd.DataFrame(prediction_proba, columns=label_encoder.classes_)
            st.write(prob_df.T)
except Exception as e:
    st.error(f"An error occurred: {e}")