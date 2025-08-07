from .base import MarketInfoBase
from core.models import CryptoInfo, Crypto
from django.db.models import OuterRef, Subquery

class BTCDominanceInfo(MarketInfoBase):
    def compute(self):
        # Récupérer toutes les cryptos suivies
        cryptos = Crypto.objects.all()

        total_market_cap = 0
        btc_market_cap = 0

        for crypto in cryptos:
            # Sous-requête pour choper la dernière info disponible pour chaque crypto
            latest_info = CryptoInfo.objects.filter(
                crypto=crypto
            ).order_by('-timestamp').first()

            if latest_info and latest_info.market_cap:
                total_market_cap += latest_info.market_cap
                if crypto.symbol.upper() == "BTC":
                    btc_market_cap = latest_info.market_cap

        if total_market_cap == 0:
            self._dominance = 0
        else:
            self._dominance = (btc_market_cap / total_market_cap) * 100

        return f"{self._dominance:.2f}%"

    def get_flag(self):
        if self._dominance >= 55:
            return 5
        elif self._dominance >= 50:
            return 4
        elif self._dominance >= 45:
            return 3
        elif self._dominance >= 40:
            return 2
        return 1

    def get_label(self):
        return "Dominance BTC"

    def get_message(self):
        if self._dominance >= 55:
            return "Bitcoin domine largement le marché."
        elif self._dominance >= 50:
            return "Forte domination de Bitcoin."
        elif self._dominance >= 45:
            return "Bitcoin reste le leader incontesté."
        elif self._dominance >= 40:
            return "Domination raisonnable du BTC."
        else:
            return "Le marché devient plus diversifié."
