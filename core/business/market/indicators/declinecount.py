from .base import MarketInfoBase
from core.models import CryptoInfo
from django.utils.timezone import now, timedelta

class DeclineCountInfo(MarketInfoBase):
    THRESHOLD = 10  # % de chute pour être considéré comme "forte chute"

    def compute(self):
        time_now = now()
        time_24h_ago = time_now - timedelta(hours=24)

        self._decline_count = 0

        for crypto in self.crypto_queryset:
            infos = (
                CryptoInfo.objects
                .filter(crypto=crypto, timestamp__range=(time_24h_ago, time_now))
                .order_by("timestamp")
            )

            if infos.count() >= 2:
                first_price = infos.first().current_price
                last_price = infos.last().current_price

                if first_price and last_price and first_price > 0:
                    change_pct = ((first_price - last_price) / first_price) * 100
                    if change_pct >= self.THRESHOLD:
                        self._decline_count += 1

        return str(self._decline_count)

    def get_flag(self):
        if self._decline_count > 20:
            return 5
        elif self._decline_count > 10:
            return 4
        elif self._decline_count > 5:
            return 3
        elif self._decline_count > 2:
            return 2
        return 1

    def get_label(self):
        return f"Chutes > {self.THRESHOLD}%"

    def get_message(self):
        if self._decline_count == 0:
            return "Aucune crypto en forte chute."
        elif self._decline_count <= 2:
            return "Très peu de cryptos en forte baisse."
        elif self._decline_count <= 5:
            return "Quelques cryptos montrent une chute significative."
        elif self._decline_count <= 10:
            return "Plusieurs cryptos sont en forte baisse."
        elif self._decline_count <= 20:
            return "Nombre important de cryptos en forte chute."
        return "Chute généralisée sur le marché crypto."
