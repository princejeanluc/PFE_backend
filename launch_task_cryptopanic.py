from shared.coingecko import fetch_and_store_crypto_data
from shared.cryptopanic import fetch_news_from_cryptopanic
from shared.db import init_django
init_django()

if __name__ =="__main__":
    fetch_and_store_crypto_data()
    fetch_news_from_cryptopanic()