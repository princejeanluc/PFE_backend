from core.models import CryptoInfo
from .base import MarketInfoBase
from datetime import datetime, timedelta

class MCDIIndicator(MarketInfoBase):
    def __init__(self, crypto_queryset):
        super().__init__(crypto_queryset)
        self.top_n = 5  # configurable si besoin
        self.result = None  # on stocke le résultat pour l'utiliser dans get_flag et get_message

    def compute(self):
        # Récupérer les dernières données (par exemple aujourd'hui ou hier)
        latest_date = CryptoInfo.objects.latest("timestamp").timestamp.date()
        data = CryptoInfo.objects.filter(timestamp__date=latest_date)

        total_market_cap = sum(item.market_cap for item in data if item.market_cap)
        top_cryptos = sorted(data, key=lambda x: x.market_cap or 0, reverse=True)[:self.top_n]
        top_market_cap = sum(item.market_cap for item in top_cryptos if item.market_cap)

        if total_market_cap == 0:
            self.result = None
        else:
            self.result = round((top_market_cap / total_market_cap) * 100, 2)

        return self.result

    def get_flag(self):
        if self.result is None:
            return 0
        if self.result >= 80:
            return 5  # Très concentré
        elif self.result >= 65:
            return 4  # Concentré
        elif self.result >= 50:
            return 3  # Moyennement concentré
        elif self.result >= 35:
            return 2  # Légèrement concentré
        else:
            return 1  # Très diversifié

    def get_label(self):
        return "MCDI"

    def get_message(self):
        if self.result is None:
            return "La concentration du marché n'a pas pu être évaluée."

        if self.result >= 80:
            return "Le marché est très concentré autour de quelques cryptomonnaies dominantes."
        elif self.result >= 65:
            return "Le marché est concentré, dominé par les grandes capitalisations."
        elif self.result >= 50:
            return "Le marché est moyennement concentré."
        elif self.result >= 35:
            return "Le marché est légèrement concentré, avec une certaine diversité."
        else:
            return "Le marché est bien diversifié, aucune crypto ne domine excessivement."
