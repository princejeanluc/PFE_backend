# core/management/commands/init_top_cryptos.py

from django.core.management.base import BaseCommand
from core.models import Crypto
from dotenv import load_dotenv
import requests
import os 

load_dotenv()

class Command(BaseCommand):
    help = 'Initialise les 20 premières cryptos par capitalisation dans la base de données'

    def handle(self, *args, **kwargs):
        url = "https://api.coingecko.com/api/v3/coins/markets"
        headers = {"x-cg-demo-api-key":os.getenv('API_CoinGecko'), "accept": "application/json"}
        params = {"vs_currency": "usd", "order": "market_cap_desc", "per_page": 20, "page": 1}
        response = requests.get(url, params=params, headers=headers)
        data = response.json()

        for coin in data:
            obj, created = Crypto.objects.get_or_create(
                id=coin["id"],
                name= coin["name"],
                symbol= coin["symbol"],
                image_url= coin["image"],
                slug= coin["id"]
            )
            if created:
                self.stdout.write(self.style.SUCCESS(f"Ajouté: {coin['name']}"))
            else:
                self.stdout.write(self.style.WARNING(f"Déjà présent: {coin['name']}"))
