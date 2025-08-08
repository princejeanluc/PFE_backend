from django.db.models import OuterRef, Subquery, F
from .base import MarketInfoBase
from core.models import CryptoInfo, MarketIndicatorSnapshot

from django.utils.timezone import now, timedelta
import numpy as np

class TopCryptoVariationInfo(MarketInfoBase):
    WINDOW_HOURS = 24  # fenêtre de comparaison

    def compute(self):
        end_time = now()
        start_time = end_time - timedelta(hours=self.WINDOW_HOURS)

        # Sous-requête: dernière info par crypto (prix + market cap)
        latest_info_sq = (
            CryptoInfo.objects
            .filter(crypto=OuterRef('pk'))
            .order_by('-timestamp')
        )

        # Sous-requête: prix le plus récent AVANT OU ÉGAL à start_time (prix à T-24h)
        past_price_sq = (
            CryptoInfo.objects
            .filter(crypto=OuterRef('pk'), timestamp__lte=start_time)
            .order_by('-timestamp')
            .values('current_price')[:1]
        )

        # Annoter le queryset fourni: market cap/price récents + prix passé
        qs = (
            self.crypto_queryset
            .annotate(
                latest_market_cap=Subquery(latest_info_sq.values('market_cap')[:1]),
                latest_price=Subquery(latest_info_sq.values('current_price')[:1]),
                past_price=Subquery(past_price_sq),
                sym=F('symbol'),
            )
            .filter(latest_market_cap__isnull=False, latest_price__isnull=False)
        ).order_by('-latest_market_cap')[:10]  # top 10 par market cap

        # Variations (dernier prix vs prix à T-24h)
        variations = []
        for c in qs:
            pp = c.past_price
            lp = c.latest_price
            if pp and lp and pp > 0:
                variations.append((lp - pp) / pp)

        if not variations:
            self._value = "N/A"
            self._numeric = None
            return self._value

        avg_var = float(np.mean(variations))
        self._numeric = round(avg_var, 4)         # ex: 0.0432
        self._value = f"{avg_var:.2%}"            # ex: "4.32%"
        return self._value

    def get_flag(self):
        # 1 = pire (baisse marquée), 5 = meilleur (hausse forte)
        if self._numeric is None:
            return 3  # neutre si inconnu
        v = self._numeric
        if v >= 0.10:
            return 5
        elif v >= 0.05:
            return 4
        elif v > -0.05:
            return 3
        elif v > -0.10:
            return 2
        else:
            return 1

    def get_label(self):
        return "Variation 24h (Top 10)"

    def get_message(self):
        if self._numeric is None:
            return "Variation indisponible pour le top 10."
        v = self._numeric
        if v >= 0.10:
            return "Croissance forte du top 10 : tendance haussière nette."
        elif v >= 0.05:
            return "Croissance modérée du top 10 : tendance haussière."
        elif v > -0.05:
            return "Variation modérée du top 10 sur 24h."
        elif v > -0.10:
            return "Baisse modérée du top 10."
        else:
            return "Baisse marquée du top 10."

    def save_snapshot(self):
        if not hasattr(self, "_value"):
            self.compute()
        MarketIndicatorSnapshot.objects.update_or_create(
            name=self.get_label(),
            defaults={
                "value": self._value,
                "numeric_value": self._numeric,
                "flag": self.get_flag(),
                "message": self.get_message(),
            }
        )
