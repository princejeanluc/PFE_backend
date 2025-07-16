import logging
import azure.functions as func
from shared.coingecko import fetch_and_store_crypto_data

app = func.FunctionApp()

@app.timer_trigger(schedule="0 0/30 * * * *", arg_name="myTimer", run_on_startup=True,
              use_monitor=False) 
def coingecko_trigger(myTimer: func.TimerRequest) -> None:
    if myTimer.past_due:
        logging.info('The timer is past due!')
    logging.info('Python timer trigger function executed.')
    # init_django()
    logging.info("⏱️ Lancement de la mise à jour horaire des cryptos.")
    fetch_and_store_crypto_data()
    logging.info("✅ Mise à jour terminée.")