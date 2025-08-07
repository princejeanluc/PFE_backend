from .base import MarketInfoBase
from core.models import CryptoInfo, MarketIndicatorSnapshot
from django.utils.timezone import now, timedelta

class DeclineCountInfo(MarketInfoBase):
    THRESHOLD = 10  # % de chute pour être considéré comme "forte chute"

    def compute(self):
        time_now = now()
        time_24h_ago = time_now - timedelta(hours=24)

        decline_count = 0

        for crypto in self.crypto_queryset:
            infos = (
                CryptoInfo.objects
                .filter(crypto=crypto, timestamp__range=(time_24h_ago, time_now))
                .order_by("timestamp")
            )

            if infos.exists():
                first = infos.first()
                last = infos.last()

                if first and last and first.current_price and last.current_price and first.current_price > 0:
                    change_pct = ((first.current_price - last.current_price) / first.current_price) * 100
                    if change_pct >= self.THRESHOLD:
                        decline_count += 1

        self._decline_count = decline_count
        self._numeric = decline_count
        self._value = str(decline_count)
        return self._value

    def get_flag(self):
        if not hasattr(self, "_decline_count"):
            self.compute()

        count = self._decline_count
        if count > 20:
            return 1  # très mauvaise situation
        elif count > 10:
            return 2
        elif count > 5:
            return 3
        elif count > 2:
            return 4
        return 5  # bon : très peu de chutes

    def get_label(self):
        return f"Chutes > {self.THRESHOLD}%"

    def get_message(self):
        count = getattr(self, "_decline_count", None)
        if count is None:
            return "Aucune donnée disponible."
        elif count == 0:
            return "Aucune crypto en forte chute."
        elif count <= 2:
            return "Très peu de cryptos en forte baisse."
        elif count <= 5:
            return "Quelques cryptos montrent une chute significative."
        elif count <= 10:
            return "Plusieurs cryptos sont en forte baisse."
        elif count <= 20:
            return "Nombre important de cryptos en forte chute."
        return "Chute généralisée sur le marché crypto."

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
