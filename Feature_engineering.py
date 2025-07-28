import requests
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import xgboost
from azure.storage.blob import BlobServiceClient
import io
import os
import json
from datetime import datetime
import numpy as np
from sqlalchemy import create_engine
import urllib

# ENV VAR from Azure App Settings
connection_string = "DefaultEndpointsProtocol=https;AccountName=cryptomodel;AccountKey=D9g+2y6lewuPErJCyyDH5fSqtrygF416RNycECZEntfcT//ALGNzdgMgfisFsBbjamn7rJoqd8F5+AStbMSsXQ==;EndpointSuffix=core.windows.net"
#os.getenv("AZURE_STORAGE_CONNECTION_STRING")
container_name = "model"

# Connect to the blob service
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# Helper to load blob content into memory
def load_blob_to_memory(blob_name):
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    return blob_client.download_blob().readall()

# Load model and encoder from blob into memory
model_blob = load_blob_to_memory("1h_prediction_Stable_xgb_regression_model.pkl")
encoder_blob = load_blob_to_memory("1h_prediction_symbol_label_encoder.pkl")
features_blob = load_blob_to_memory("1h_prediction_Stable_final_used_features.csv")

# Load into objects
model = joblib.load(io.BytesIO(model_blob))
encode_path = joblib.load(io.BytesIO(encoder_blob))
expected_features = pd.read_csv(io.BytesIO(features_blob))["0"].tolist()

# Constants
LOOKBACK_HOURS = 504  # 3 weeks
BASE_URL = "https://cryptocurrency.azurewebsites.net/api"
HEADERS = {"Accept": "application/json"}

# Fetch data from API
df_coin = pd.DataFrame(requests.get(f"{BASE_URL}/CoinData?lookbackHours={LOOKBACK_HOURS}", headers=HEADERS).json())
df_investor = pd.DataFrame(requests.get(f"{BASE_URL}/InvestorGrade?lookbackHours={LOOKBACK_HOURS}", headers=HEADERS).json())
df_sentiment = pd.DataFrame(requests.get(f"{BASE_URL}/Sentiment?lookbackHours={LOOKBACK_HOURS}", headers=HEADERS).json())

# Convert date columns to datetime
df_coin["date"] = pd.to_datetime(df_coin["date"])
df_investor["date"] = pd.to_datetime(df_investor["date"])
df_sentiment["date"] = pd.to_datetime(df_sentiment["date"])

# Floor InvestorGrade date to day
df_investor["date"] = df_investor["date"].dt.floor("D")

# Preserve hourly date for coin and sentiment, but also floor to join with investor
df_coin["dateHour"] = df_coin["date"]
df_sentiment["dateHour"] = df_sentiment["date"]
df_coin["date"] = df_coin["date"].dt.floor("D")
df_sentiment["date"] = df_sentiment["date"].dt.floor("D")

# Merge coin + investor on Symbol + floored date
df = df_coin.merge(df_investor, on=["symbol", "date"], how="left")

# Merge in sentiment on Symbol + exact hour
df = df.merge(df_sentiment, left_on=["symbol", "dateHour"], right_on=["symbol", "dateHour"], how="inner")

# Use full datetime again
df["date"] = df["dateHour"]
df.drop(columns=["dateHour"], inplace=True)

#--- Feature engineering Start ---
df["sentimentScore"] = df["positiveReddit"] - df["negativeReddit"]

# Lag Features
for col in ["price", "volume24h", "volumeChange24h", "fearandGreed", "sentimentScore"]:
    df[f"{col}_lag_1"] = df.groupby("symbol")[col].shift(1)
    df[f"{col}_lag_2"] = df.groupby("symbol")[col].shift(2)

# Rolling Features
df["price_rolling_mean_3h"] = df.groupby("symbol")["price"].transform(lambda x: x.rolling(3).mean())
df["volume_rolling_std_6h"] = df.groupby("symbol")["volume24h"].transform(lambda x: x.rolling(6).std())
df["sentiment_rolling_mean_3h"] = df.groupby("symbol")["sentimentScore"].transform(lambda x: x.rolling(3).mean())

# Momentum Features
df["price_momentum_1h"] = df["price"] - df["price_lag_1"]
df["sentiment_momentum_1h"] = df["sentimentScore"] - df["sentimentScore_lag_1"]

# --- Technical indicators ---
# Calulate Momentum 
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_histogram = macd - macd_signal
    return macd, macd_signal, macd_histogram

# Compute RSI and MACD per symbol
df = df.sort_values(by=["symbol", "date"])  # Ensure it's sorted correctly

df["RSI"] = df.groupby("symbol")["price"].transform(lambda x: compute_rsi(x, period=14))
df["MACD"], df["MACD_Signal"], df["MACD_Histogram"] = zip(*df.groupby("symbol")["price"].transform(lambda x: pd.DataFrame(compute_macd(x)).T.values.tolist()))
# Momentum Confirmation Logic
df["MomentumScore"] = (df["percentChange1h"] + df["percentChange24h"]) * np.log1p(df["volumeChange24h"])
df["MomentumConfirmed"] = (
    (df["RSI"] > 55) &
    (df["MACD_Histogram"] > 0) &
    (df["percentChange24h"] > 0) &
    (df["volumeChange24h"] > 0) &
    (df["tmTraderGrade24hPctChange"] > 0)
)

# --- Bollinger Bands Calculation ---
def compute_bollinger_bands(series, window=20, num_std=2):
    ma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    width = upper - lower
    return upper, lower, width

# Compute for each symbol
df = df.sort_values(by=["symbol", "date"])
df["bb_upper"], df["bb_lower"], df["bb_width"] = zip(*df.groupby("symbol")["price"].transform(
    lambda x: pd.DataFrame(compute_bollinger_bands(x)).T.values.tolist()
))

# Flag: Squeeze = Band width at 20-period low
df["bollingerBandSqueeze"] = df.groupby("symbol")["bb_width"].transform(
    lambda x: x == x.rolling(20).min()
)

# Flag: Breakout (up or down)
df["bollingerBandBreakout"] = (
    (df["price"] > df["bb_upper"]) | 
    (df["price"] < df["bb_lower"])
)

# Mean Reversion in Extremes
# Required: rolling mean already exists
if "price_rolling_mean_3h" not in df.columns:
    df["price_rolling_mean_3h"] = df.groupby("symbol")["price"].transform(lambda x: x.rolling(3).mean())

# Distance from rolling mean
df["price_deviation_pct"] = (df["price"] - df["price_rolling_mean_3h"]) / df["price_rolling_mean_3h"]

# --- Mean Reversion Long Signal (Oversold) ---
df["meanReversionLongSignal"] = (
    (df["RSI"] < 40) & 
    (df["price_deviation_pct"] < -0.05)  # 5% below mean
)

# --- Mean Reversion Short Signal (Overbought) ---
df["meanReversionShortSignal"] = (
    (df["RSI"] > 60) &
    (df["price_deviation_pct"] > 0.05)   # 5% above mean
)

#Volume Spike
# Compute 24h rolling volume mean per symbol
df["volume24h_rolling_mean_24h"] = df.groupby("symbol")["volume24h"].transform(lambda x: x.rolling(24).mean())

# Compute spike ratio
df["volume_spike_ratio"] = df["volume24h"] / df["volume24h_rolling_mean_24h"]

# Flag volume spike when it's 1.5x or more than the rolling average
df["volumeSpikeSignal"] = df["volume_spike_ratio"] > 1.5

#Multi-Factor Confluence Score
df["multiFactorConfluenceSignal"] = (
    (df["bollingerBandBreakout"] == True) &
    (df["volumeSpikeSignal"] == True) &
    (df["RSI"] > 50) &
    (df["tmTraderGrade"] >= 70) &        # Adjust based on real value ranges
    (df["quantGrade"] >= 60) & 
    (df["sentiment_momentum_1h"] > 0)    # Improving sentiment
)


# --- Models Start --
# 1h_prediction_Stable
# Only use expected features
# Encode symbol
df["symbol_encoded"] = encode_path.transform(df["symbol"])

# Get the latest record by date for each symbol
df = (
    df.sort_values(by=["symbol_encoded", "date"])
    .groupby("symbol_encoded")
    .last()
    .reset_index()
)

df_predict = df[expected_features]

# Predict
prediction = model.predict(df_predict)

# Add predictions to DataFrame
df["Predicted_Price_1h_Stable"] = prediction

# Inverse transform symbol encoding
df["symbol"] = encode_path.inverse_transform(df["symbol_encoded"])

# Replace NaNs with None
df = df.replace({np.nan: None})

# Define engineered features (including prediction)
engineered_features = [
    "sentimentScore",
    "price_lag_1", "price_lag_2",
    "volume24h_lag_1", "volume24h_lag_2",
    "volumeChange24h_lag_1", "volumeChange24h_lag_2",
    "fearandGreed_lag_1", "fearandGreed_lag_2",
    "sentimentScore_lag_1", "sentimentScore_lag_2",
    "price_rolling_mean_3h", "volume_rolling_std_6h", "sentiment_rolling_mean_3h",
    "price_momentum_1h", "sentiment_momentum_1h",
    "Predicted_Price_1h_Stable", "MomentumScore", "MomentumConfirmed", "bollingerBandSqueeze", "bollingerBandBreakout",
    "meanReversionLongSignal", "meanReversionShortSignal","price_deviation_pct","volume24h_rolling_mean_24h",
    "volume_spike_ratio", "volumeSpikeSignal", "multiFactorConfluenceSignal"
]

# --- Post Processing ---
# Post each row with engineered features
url = "https://cryptocurrency.azurewebsites.net/api/FeatureEngineering"

for _, row in df.iterrows():
    payload = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S") + row["symbol"],
        "symbol": row["symbol"],
        "date": row["date"].isoformat() if pd.notnull(row["date"]) else None,
    }

    for col in engineered_features:
        val = row.get(col)
        payload[col] = val.isoformat() if isinstance(val, pd.Timestamp) else val
    est_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
    if est_hour.isoformat() == row["date"]:
        response = requests.post(url, json=payload)
        print(f"Posted {row['symbol']} | Status: {response.status_code}")
    else:
        print(est_hour.isoformat())
        print(row["date"])
        print("Date does not match current hour, skipping post for", row["symbol"])
