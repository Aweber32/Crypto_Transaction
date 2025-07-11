import logging
import azure.functions as func
import json

def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        from .Stable_future_price_prediction import run as stable_run
        stable_run()
        print("CryptoTransaction function executed successfully.")
        return func.HttpResponse("CryptoTransaction executed.", status_code=200)
    except Exception as e:
        return func.HttpResponse(f"Error occurred: {e}", status_code=500)
