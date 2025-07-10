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
model_blob = load_blob_to_memory("Stable_xgb_regression_model.pkl")
encoder_blob = load_blob_to_memory("symbol_label_encoder.pkl")
features_blob = load_blob_to_memory("Stable_final_used_features.csv")

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
df_transaction = pd.DataFrame(requests.get(f"{BASE_URL}/Transaction?lookbackHours={LOOKBACK_HOURS}", headers=HEADERS).json())

# Convert date columns to datetime
df_coin["date"] = pd.to_datetime(df_coin["date"])
df_investor["date"] = pd.to_datetime(df_investor["date"])
df_sentiment["date"] = pd.to_datetime(df_sentiment["date"])
df_transaction["date"] = pd.to_datetime(df_transaction["date"])

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

# Feature engineering
df["sentimentScore"] = df["positiveReddit"] - df["negativeReddit"]
# Lag Features
for col in ["price", "volume24h", "volumeChange24h", "fearandGreed", "sentimentScore"]:
    df[f"{col}_lag_1"] = df.groupby("symbol")[col].shift(1)
    df[f"{col}_lag_2"] = df.groupby("symbol")[col].shift(2)
# Rolling Features
df["price_rolling_mean_3h"] = df.groupby("symbol")["price"].transform(lambda x: x.rolling(3).mean())
df["volume_rolling_std_6h"] = df.groupby("symbol")["volume24h"].transform(lambda x: x.rolling(6).std())
df["sentiment_rolling_mean_3h"] = df.groupby("symbol")["sentimentScore"].transform(lambda x: x.rolling(3).mean())
#Momentum Features
df["price_momentum_1h"] = df["price"] - df["price_lag_1"]
df["sentiment_momentum_1h"] = df["sentimentScore"] - df["sentimentScore_lag_1"]
# Load In le encoder for symbols Text
df["symbol_encoded"] = encode_path.transform(df["symbol"])
# Save full history before collapsing
df_history = df.copy()
# Merge transaction data into history to get historical buy/sell/hold signals
df_history = df_history.merge(
    df_transaction[["symbol", "date", "buy_Sell_Hold_Skip"]],
    on=["symbol", "date"],
    how="left"
)
#get the latest record by date for each symbol
df = (
    df.sort_values(by=["symbol_encoded", "date"])
      .groupby("symbol_encoded")
      .last()
      .reset_index()
)
# Only use expected features
df_predict = df[expected_features]

prediction = model.predict(df_predict)
# Prepare the final DataFrame with predictions
# Inverse transform the symbol_encoded column to get the original symbol names
df["symbol"] = encode_path.inverse_transform(df["symbol_encoded"])

# Add predictions to DataFrame
df["predicted_pct_change"] = prediction

# ---- BUY / SELL / HOLD LOGIC ----
def compute_decision(symbol, current_pred, df_history):
    df_symbol = df_history[df_history["symbol"] == symbol].sort_values("date", ascending=False)

    if df_symbol.empty:
        return "buy" if current_pred > 0 else "skip"

    # Get most recent historical decision if it exists
    prev_action_row = df_symbol[df_symbol["buy_Sell_Hold_Skip"].notna()]
    prev_action = prev_action_row.iloc[0]["buy_Sell_Hold_Skip"] if not prev_action_row.empty else None

    if prev_action is None:
        return "buy" if current_pred > 0 else "skip"

    if prev_action in ["buy", "hold"]:
        total_change = 0
        for _, row in df_symbol.iterrows():
            action = row.get("buy_Sell_Hold_Skip")
            if action not in ["buy", "hold"]:
                break
            total_change += row.get("percentChange1h", 0)
        total_change += current_pred

        if total_change < -1:
            return "sell"
        elif total_change > 2:
            return "sell"
        else:
            return "hold"

    if prev_action in ["sell", "skip"]:
        return "buy" if current_pred > 0 else "skip"

    return "skip"



# Apply logic
df["buy_Sell_Hold_Skip"] = df.apply(
    lambda row: compute_decision(row["symbol"], row["predicted_pct_change"], df_history),
    axis=1
)

# Final output
final_output = df[["symbol", "date", "price", "predicted_pct_change", "buy_Sell_Hold_Skip"]]

# API URL
url = "https://cryptocurrency.azurewebsites.net/api/Transaction"

# Loop through each row
for _, row in df.iterrows():
    payload = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S") + row["symbol"],
        "symbol": row["symbol"],
        "date": row["date"].isoformat(),  # Convert datetime to string
        "price": row["price"],
        "Predicted_Price_1h": row["predicted_pct_change"],
        "buy_Sell_Hold_Skip": row["buy_Sell_Hold_Skip"]
    }

    response = requests.post(url, json=payload)

    print(f"Response for {row['symbol']} on {row['date']}: {response.status_code}")

