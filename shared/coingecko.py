import requests
from django.utils.timezone import make_aware
from datetime import datetime
from django.utils.timezone import is_naive
from django.utils.dateparse import parse_datetime
from dotenv import load_dotenv
load_dotenv()

import os
import sys
from .db import init_django
# 1. Ajouter POSA_backend au PYTHONPATH
# BACKEND_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'POSA_backend'))
# sys.path.append(BACKEND_PATH)

# 2. Définir le module de configuration Django
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', os.getenv("DJANGO_SETTINGS_MODULE"))  # ← ici on utilise config.settings

# 3. Initialiser Django
# import django
# django.setup()

init_django()
from core.models import Crypto, CryptoInfo



def ensure_aware(dt):
    if dt is None:
        return None
    return make_aware(dt) if is_naive(dt) else dt


def fetch_and_store_crypto_data():
    headers = {
        "x-cg-demo-api-key": os.getenv("API_CoinGecko"),
        "accept": "application/json"
    }

    # 1. Préparer la liste des slugs
    slugs = [crypto.slug for crypto in Crypto.objects.all()]
    ids_param = ','.join(slugs)

    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "ids": ids_param,
        "order": "market_cap_desc",
        "per_page": 250,
        "page": 1,
        "price_change_percentage": "24h"
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        results = response.json()

        for item in results:
            slug = item["id"]
            try:
                crypto = Crypto.objects.get(slug=slug)
            except Crypto.DoesNotExist:
                print(f"Crypto non trouvée : {slug}")
                continue

            # Convertir les timestamps ISO 8601
            last_updated = ensure_aware(parse_datetime(item.get("last_updated")))
            ath_date = ensure_aware(parse_datetime(item.get("ath_date")))
            atl_date = ensure_aware(parse_datetime(item.get("atl_date")))

            # Empêche les doublons
            if CryptoInfo.objects.filter(crypto=crypto, timestamp=last_updated).exists():
                print(f"Déjà présent : {crypto.symbol} @ {last_updated}")
                continue

            CryptoInfo.objects.create(
                crypto=crypto,
                timestamp=last_updated,
                current_price=item.get("current_price"),
                market_cap=item.get("market_cap"),
                market_cap_rank=item.get("market_cap_rank"),
                fully_diluted_valuation=item.get("fully_diluted_valuation"),
                total_volume=item.get("total_volume"),
                high_24h=item.get("high_24h"),
                low_24h=item.get("low_24h"),
                price_change_24h=item.get("price_change_24h"),
                price_change_percentage_24h=item.get("price_change_percentage_24h"),
                market_cap_change_24h=item.get("market_cap_change_24h"),
                market_cap_change_percentage_24h=item.get("market_cap_change_percentage_24h"),
                circulating_supply=item.get("circulating_supply"),
                total_supply=item.get("total_supply"),
                max_supply=item.get("max_supply"),
                ath=item.get("ath"),
                ath_change_percentage=item.get("ath_change_percentage"),
                ath_date=ath_date,
                atl=item.get("atl"),
                atl_change_percentage=item.get("atl_change_percentage"),
                atl_date=atl_date,
                last_updated=last_updated,
                # Tu pourras ajouter drawdown_from_ath plus tard via une fonction
            )
            print(f"Ajouté : {crypto.symbol.upper()} @ {last_updated}")

    except Exception as e:
        print(f"⚠️ Erreur de récupération : {e}")
