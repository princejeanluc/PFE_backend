from django.db.models import OuterRef, Subquery
from .base import MarketInfoBase
from core.models import CryptoInfo, MarketIndicatorSnapshot
from django.utils.timezone import now, timedelta

class UpwardTrendInfo(MarketInfoBase):
    WINDOW_HOURS = 24  # fenêtre de comparaison

    def compute(self):
        t_end = now()
        t_start = t_end - timedelta(hours=self.WINDOW_HOURS)

        # Dernier prix avant/à t_start (prix de référence T-24h)
        past_price_sq = (
            CryptoInfo.objects
            .filter(crypto=OuterRef('pk'), timestamp__lte=t_start)
            .order_by('-timestamp')
            .values('current_price')[:1]
        )

        # Dernier prix avant/à t_end (prix courant)
        latest_price_sq = (
            CryptoInfo.objects
            .filter(crypto=OuterRef('pk'), timestamp__lte=t_end)
            .order_by('-timestamp')
            .values('current_price')[:1]
        )

        # Une seule requête: on annote chaque crypto avec past_price et latest_price
        qs = (
            self.crypto_queryset
            .annotate(
                past_price=Subquery(past_price_sq),
                latest_price=Subquery(latest_price_sq),
            )
            .filter(past_price__isnull=False, latest_price__isnull=False)
        )

        count_up = 0
        count_total = 0
        for c in qs:
            pp = c.past_price
            lp = c.latest_price
            if pp and lp:  # pp > 0 pas nécessaire pour une simple comparaison de tendance
                if lp > pp:
                    count_up += 1
                count_total += 1

        if count_total == 0:
            self._percentage_up = None
            self._numeric = None
            self._value = "N/A"
            return self._value

        percentage = (count_up / count_total) * 100
        self._percentage_up = percentage
        self._numeric = percentage
        self._value = f"{percentage:.1f}%"
        return self._value

    def get_flag(self):
        # 1 = pire (majorité en baisse), 5 = meilleur (majorité en forte hausse)
        p = getattr(self, "_percentage_up", None)
        if p is None:
            return 3  # neutre si inconnu
        if p > 75:
            return 5
        elif p > 50:
            return 4
        elif p > 30:
            return 3
        elif p > 10:
            return 2
        return 1

    def get_label(self):
        return "Cryptos en hausse"

    def get_message(self):
        p = getattr(self, "_percentage_up", None)
        if p is None:
            return "Données insuffisantes pour calculer la tendance."
        if p > 75:
            return "Majorité des cryptos en forte hausse."
        elif p > 50:
            return "Tendance haussière générale sur le marché."
        elif p > 30:
            return "Certaines cryptos montrent des signes positifs."
        elif p > 10:
            return "Le marché reste globalement neutre."
        return "Majorité des cryptos en baisse."

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
