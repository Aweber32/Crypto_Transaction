import logging
import azure.functions as func
import json

def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        data = req.get_json()
        symbol = data.get("symbol")
        prediction = data.get("predicted_pct_change")
        price = data.get("price")

        logging.info(f"Transaction received for {symbol}: predicted {prediction}, price {price}")

        # Perform your transaction logic here

        return func.HttpResponse("Transaction processed", status_code=200)
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return func.HttpResponse("Error processing transaction", status_code=500)
