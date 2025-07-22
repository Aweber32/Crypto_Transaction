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

# ENV VAR from Azure App Settings
connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
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

# Encode symbol
df["symbol_encoded"] = encode_path.transform(df["symbol"])

# Get the latest record by date for each symbol
df = (
    df.sort_values(by=["symbol_encoded", "date"])
    .groupby("symbol_encoded")
    .last()
    .reset_index()
)

# --- Models Start --
# 1h_prediction_Stable
# Only use expected features
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
    "Predicted_Price_1h_Stable"
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

    response = requests.post(url, json=payload)
    print(f"Posted {row['symbol']} | Status: {response.status_code}")
