# syntax=docker/dockerfile:1

# Use official slim Python 3.12 base image
FROM python:3.12-slim-bookworm

ENV AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=cryptomodel;AccountKey=D9g+2y6lewuPErJCyyDH5fSqtrygF416RNycECZEntfcT//ALGNzdgMgfisFsBbjamn7rJoqd8F5+AStbMSsXQ==;EndpointSuffix=core.windows.net"

# Set working directory
WORKDIR /app

# Copy the script into the container
COPY Stable_future_price_prediction.py ./

# (Optional) Install dependencies if you have a requirements.txt
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Define the default command to run your script
CMD ["python", "Stable_future_price_prediction.py"]
