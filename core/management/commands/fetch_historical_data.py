# core/management/commands/fetch_historical_data.py

from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from core.models import Crypto, CryptoInfo
from dotenv import load_dotenv
import os
import requests
import time
from datetime import datetime, timedelta , timezone

load_dotenv()

API_KEY = os.getenv("API_CoinGecko")
HEADERS = {"x-cg-demo-api-key": API_KEY, "accept": "application/json"}

class Command(BaseCommand):
    help = "Récupère les données historiques (1h) sur 1 an pour les cryptos déjà enregistrées"

    def handle(self, *args, **kwargs):
        cryptos = Crypto.objects.all()
        for crypto in cryptos:
            self.stdout.write(f"\n🔄 Traitement de {crypto.name} ({crypto.slug})")
            try:
                self.fetch_and_store_crypto_info(crypto)
            except Exception as e:
                self.stderr.write(f"⚠️ Erreur pour {crypto.name}: {e}")
                continue

    def fetch_and_store_crypto_info(self, crypto):
        time_ranges = self.get_time_ranges(days=360, chunk_days=90)
        for from_ts, to_ts in time_ranges:
            url = f"https://api.coingecko.com/api/v3/coins/{crypto.slug}/market_chart/range"
            params = {"vs_currency": "usd", "from": from_ts, "to": to_ts}
            response = requests.get(url, params=params, headers=HEADERS)
            if response.status_code != 200:
                raise Exception(f"Erreur API ({response.status_code}): {response.text}")
            data = response.json()
            self.save_market_data(crypto, data)
            time.sleep(1.5)  # éviter d'être bloqué

    def get_time_ranges(self, days=360, chunk_days=90):
        end = datetime.now(tz=timezone.utc)
        start = end - timedelta(days=days)
        ranges = []
        while start < end:
            segment_end = min(start + timedelta(days=chunk_days), end)
            ranges.append((int(start.timestamp()), int(segment_end.timestamp())))
            start = segment_end
        return ranges

    def save_market_data(self, crypto, data):
        prices = data.get("prices", [])
        market_caps = {int(p[0]): p[1] for p in data.get("market_caps", [])}
        total_volumes = {int(p[0]): p[1] for p in data.get("total_volumes", [])}

        created_count = 0
        for point in prices:
            ts_ms, price = point
            ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)

            if CryptoInfo.objects.filter(crypto=crypto, timestamp=ts).exists():
                continue

            CryptoInfo.objects.create(
                crypto=crypto,
                timestamp=ts,
                current_price=price,
                market_cap=market_caps.get(ts_ms),
                total_volume=total_volumes.get(ts_ms),
                # Tu peux ajouter d'autres valeurs pré-remplies ou calculées ici si nécessaire
            )
            created_count += 1

        self.stdout.write(self.style.SUCCESS(f"✅ Ajoutés: {created_count} entrées pour {crypto.name}"))
