from .base import MarketInfoBase
from core.models import CryptoInfo
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
            if infos.count() >= 2:
                first_price = infos.first().current_price
                last_price = infos.last().current_price
                if first_price and last_price:
                    if last_price > first_price:
                        count_up += 1
                    count_total += 1

        if count_total == 0:
            self._percentage_up = None
            return "N/A"

        percentage = (count_up / count_total) * 100
        self._percentage_up = percentage
        return f"{percentage:.1f}%"

    def get_flag(self):
        if self._percentage_up is None:
            return 3
        if self._percentage_up > 75:
            return 5
        elif self._percentage_up > 50:
            return 4
        elif self._percentage_up > 30:
            return 3
        elif self._percentage_up > 10:
            return 2
        return 1

    def get_label(self):
        return "Cryptos en hausse"

    def get_message(self):
        if self._percentage_up is None:
            return "Données insuffisantes pour calculer la tendance."
        if self._percentage_up > 75:
            return "Majorité des cryptos en forte hausse."
        elif self._percentage_up > 50:
            return "Tendance haussière générale sur le marché."
        elif self._percentage_up > 30:
            return "Certaines cryptos montrent des signes positifs."
        elif self._percentage_up > 10:
            return "Le marché reste globalement neutre."
        return "Majorité des cryptos en baisse."
