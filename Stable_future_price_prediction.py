import requests
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import xgboost

#Below routes are used for local development
model_path = r"C:\Users\alexw\OneDrive\Desktop\Code\Crypto_Model\models\Stable_xgb_regression_model.pkl"
encode_path = r"C:\Users\alexw\OneDrive\Desktop\Code\Crypto_Model\models\symbol_label_encoder.pkl"
feature_used = r"C:\Users\alexw\OneDrive\Desktop\Code\Crypto_Model\models\Stable_final_used_features.csv"

# Constants
LOOKBACK_HOURS = 48  # 48 hours
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
encoder = joblib.load(encode_path)
df["symbol_encoded"] = encoder.transform(df["symbol"])
# Save full history before collapsing
df_history = df.copy()
#get the latest record by date for each symbol
df = (
    df.sort_values(by=["symbol_encoded", "date"])
      .groupby("symbol_encoded")
      .last()
      .reset_index()
)
# Only use expected features
expected_features = pd.read_csv(feature_used)["0"].tolist()
df_predict = df[expected_features]
# load the model and make predictions
model = joblib.load(model_path)
prediction = model.predict(df_predict)
# Prepare the final DataFrame with predictions
# Inverse transform the symbol_encoded column to get the original symbol names
df["symbol"] = encoder.inverse_transform(df["symbol_encoded"])

# Add predictions to DataFrame
df["predicted_pct_change"] = prediction

# ---- BUY / SELL / HOLD LOGIC ----
def compute_decision(symbol, current_pred, df_history):
    df_symbol = df_history[df_history["symbol"] == symbol].sort_values("date", ascending=False)
    if df_symbol.empty:
        return "buy" if current_pred > 0 else "skip"

    prev_action = df_symbol.iloc[0].get("buy_sell_hold", "skip")  # default to 'skip' if missing

    if prev_action in ["buy", "hold"]:
        total_change = 0
        for _, row in df_symbol.iterrows():
            if row.get("buy_sell_hold") not in ["buy", "hold"]:
                break
            total_change += row.get("percentChange1h", 0)
        total_change += current_pred

        if total_change < -1:
            return "sell"
        elif total_change > 2:
            return "sell"
        else:
            return "hold"
    elif prev_action == "sell" or prev_action == "skip":
        return "buy" if current_pred > 0 else "skip"

    return "skip"

# Apply logic
df["buy_sell_hold"] = df.apply(
    lambda row: compute_decision(row["symbol"], row["predicted_pct_change"], df_history),
    axis=1
)

# Final output
final_output = df[["symbol", "date", "price", "predicted_pct_change", "buy_sell_hold"]]
print(final_output)



