import logging
from shared.db import init_django
from shared.market_snapshot import make_market_snapshot
from shared.prediction import predict_all_cryptos
init_django()

if __name__ =="__main__":
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