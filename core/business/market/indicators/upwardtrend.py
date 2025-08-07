from .base import MarketInfoBase
from core.models import CryptoInfo, MarketIndicatorSnapshot
from django.utils.timezone import now, timedelta

class UpwardTrendInfo(MarketInfoBase):
    def compute(self):
        time_now = now()
        time_24h_ago = time_now - timedelta(hours=24)

        count_up = 0
        count_total = 0

        for crypto in self.crypto_queryset:
            infos = (
                CryptoInfo.objects
                .filter(crypto=crypto, timestamp__range=(time_24h_ago, time_now))
                .order_by("timestamp")
            )

            if infos.exists():
                first = infos.first()
                last = infos.last()
                if first and last and first.current_price and last.current_price:
                    if last.current_price > first.current_price:
                        count_up += 1
                    count_total += 1

        if count_total == 0:
            self._percentage_up = None
            self._value = "N/A"
            self._numeric = None
            return self._value

        percentage = (count_up / count_total) * 100
        self._percentage_up = percentage
        self._numeric = percentage
        self._value = f"{percentage:.1f}%"
        return self._value

    def get_flag(self):
        if not hasattr(self, "_percentage_up") or self._percentage_up is None:
            return 3  # neutre par défaut

        p = self._percentage_up
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
