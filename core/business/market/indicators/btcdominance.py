from .base import MarketInfoBase
from core.models import CryptoInfo, Crypto, MarketIndicatorSnapshot
from django.db.models import OuterRef, Subquery, F

class BTCDominanceInfo(MarketInfoBase):
    # Fenêtre : instantané (pas d'historique requis)
    WINDOW = None

    def compute(self):
        # Limiter aux cryptos du queryset fourni (évite de scanner tout le marché)
        crypto_ids = self.crypto_queryset.values('id')

        # Sous-requête : dernière info par crypto
        latest_info_subquery = CryptoInfo.objects.filter(
            crypto=OuterRef('pk')
        ).order_by('-timestamp')

        cryptos_with_info = (
            Crypto.objects
            .filter(id__in=crypto_ids)
            .annotate(
                latest_market_cap=Subquery(latest_info_subquery.values('market_cap')[:1]),
                latest_symbol=F('symbol')
            )
            .filter(latest_market_cap__isnull=False)
        )

        total_market_cap = 0
        btc_market_cap = 0

        for c in cryptos_with_info:
            mc = c.latest_market_cap
            total_market_cap += mc
            if c.latest_symbol.upper() == "BTC":
                btc_market_cap = mc

        if total_market_cap == 0:
            self._numeric = None
            self._value = "N/A"
        else:
            dominance = (btc_market_cap / total_market_cap) * 100
            self._numeric = round(dominance, 2)
            self._value = f"{self._numeric:.2f}%"

        return self._value

    def get_flag(self):
        # 1 = pire (marché peu diversifié si dominance élevée), 5 = meilleur
        if self._numeric is None:
            return 3
        v = self._numeric
        if v >= 55:
            return 1
        elif v >= 50:
            return 2
        elif v >= 45:
            return 3
        elif v >= 40:
            return 4
        else:
            return 5

    def get_label(self):
        return "Dominance BTC"

    def get_message(self):
        if self._numeric is None:
            return "La dominance du Bitcoin n’a pas pu être évaluée."
        v = self._numeric
        if v >= 55:
            return "Bitcoin domine largement le marché (diversification faible)."
        elif v >= 50:
            return "Forte domination de Bitcoin."
        elif v >= 45:
            return "Bitcoin reste le leader, dominance notable."
        elif v >= 40:
            return "Domination raisonnable du BTC."
        return "Marché plus diversifié : dominance BTC modérée."

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
