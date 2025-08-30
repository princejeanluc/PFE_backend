import logging
import azure.functions as func
from shared.coingecko import fetch_and_store_crypto_data
from shared.cryptopanic import fetch_news_from_cryptopanic
from shared.market_snapshot import make_market_snapshot
from shared.prediction import predict_all_cryptos
#from shared.pytrends import get_interest_crypto_from_pytrends

app = func.FunctionApp()

@app.timer_trigger(schedule="0 0/30 * * * *", arg_name="myTimer", run_on_startup=True,
              use_monitor=False) 
def action(myTimer: func.TimerRequest) -> None:
    if myTimer.past_due:
        logging.info('The timer is past due!')
    logging.info('Python timer trigger function executed.')
    try : 
        logging.info(" Lancement de la mise à jour mi-horaire des cryptos.")
        fetch_and_store_crypto_data()
        logging.info(" Fin de la mise à jour mi-horaire des cryptos")
    except Exception as e: 
        logging.info(f"Une erreur est survenue lors de la mise à jour des cryptos : {e}")
    #try  :
        #logging.info("Lancement de la mise à jour des news")
        #fetch_news_from_cryptopanic()
        #logging.info("Fin de la mise à jour des news")
    #except Exception as e: 
        #logging.info(f"Une erreur est survenue lors de la mise à jour des news : {e}")
    
    #try  :
        #logging.info("Lancement de la mise à jour de la variable exogène 'intérêt'")
        #get_interest_crypto_from_pytrends()
        #logging.info("Fin de la mise à jour de la variable exogène 'intérêt'")
    #except Exception as e: 
        #logging.info(f"Une erreur est survenue lors de la mise à jour de la variable exogène 'intérêt' : {e}")
    #logging.info(" Mise à jour terminée.")

@app.timer_trigger(schedule="0 45 * * * *", arg_name="myTimer", run_on_startup=True,
              use_monitor=False) 
def marketsnapshot(myTimer: func.TimerRequest) -> None:
    if myTimer.past_due:
        logging.info('The timer is past due!')
    logging.info('Python timer trigger function executed.')
    try  :
        logging.info("Lancement des snapshot du marché")
        make_market_snapshot()
        logging.info("Fin des snapshot du marché")
    except Exception as e: 
        logging.info(f"Une erreur est survenue lors des snapshot du marché : {e}")

    try  :
        logging.info("Prédiction du marché")
        predict_all_cryptos()
        logging.info("Fin des Prédiction du marché")
    except Exception as e: 
        logging.info(f"Une erreur est survenue lors des Prédiction du marché : {e}")