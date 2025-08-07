from .base import MarketInfoBase
from core.models import CryptoInfo, Crypto, MarketIndicatorSnapshot
from django.db.models import OuterRef, Subquery, F
from django.utils.timezone import now

class BTCDominanceInfo(MarketInfoBase):
    def compute(self):
        # Sous-requête : dernière info par crypto
        latest_info_subquery = CryptoInfo.objects.filter(
            crypto=OuterRef('pk')
        ).order_by('-timestamp')

        cryptos_with_info = Crypto.objects.annotate(
            latest_market_cap=Subquery(
                latest_info_subquery.values('market_cap')[:1]
            ),
            latest_symbol=F('symbol')
        ).filter(latest_market_cap__isnull=False)

        total_market_cap = 0
        btc_market_cap = 0

        for crypto in cryptos_with_info:
            market_cap = crypto.latest_market_cap
            total_market_cap += market_cap
            if crypto.latest_symbol.upper() == "BTC":
                btc_market_cap = market_cap

        if total_market_cap == 0:
            self._numeric = None
            self._value = "N/A"
        else:
            dominance = (btc_market_cap / total_market_cap) * 100
            self._numeric = round(dominance, 2)
            self._value = f"{self._numeric:.2f}%"

        return self._value

    def get_flag(self):
        if self._numeric is None:
            return 3
        if self._numeric >= 55:
            return 5
        elif self._numeric >= 50:
            return 4
        elif self._numeric >= 45:
            return 3
        elif self._numeric >= 40:
            return 2
        return 1

    def get_label(self):
        return "Dominance BTC"

    def get_message(self):
        if self._numeric is None:
            return "La dominance du Bitcoin n’a pas pu être évaluée."
        if self._numeric >= 55:
            return "Bitcoin domine largement le marché."
        elif self._numeric >= 50:
            return "Forte domination de Bitcoin."
        elif self._numeric >= 45:
            return "Bitcoin reste le leader incontesté."
        elif self._numeric >= 40:
            return "Domination raisonnable du BTC."
        return "Le marché devient plus diversifié."

    def save_snapshot(self):
        if not hasattr(self, "_value"):
            self.compute()

        MarketIndicatorSnapshot.objects.update_or_create(
            name=self.__class__.__name__,
            defaults={
                "value": self._value,
                "numeric_value": self._numeric,
                "flag": self.get_flag(),
                "message": self.get_message(),
            }
        )
