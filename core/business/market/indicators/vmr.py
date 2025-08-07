from core.models import CryptoInfo, MarketIndicatorSnapshot
from .base import MarketInfoBase
from django.utils.timezone import now
from django.db.models import Max
from collections import defaultdict

class VMRIndicator(MarketInfoBase):
    def __init__(self, crypto_queryset):
        super().__init__(crypto_queryset)
        self.top_n = 5

    def compute(self):
        latest_timestamp = CryptoInfo.objects.aggregate(latest=Max("timestamp"))["latest"]
        if not latest_timestamp:
            self._value = "N/A"
            self._numeric = None
            return self._value

        latest_date = latest_timestamp.date()
        data = CryptoInfo.objects.filter(timestamp__date=latest_date)

        volumes = []
        total_volume = 0

        for item in data:
            if item.total_volume:
                volumes.append((item.total_volume, item.crypto.symbol))
                total_volume += item.total_volume

        if total_volume == 0 or not volumes:
            self._value = "N/A"
            self._numeric = None
            return self._value

        top_volume = sum(v for v, _ in sorted(volumes, reverse=True)[:self.top_n])
        vmr = round((top_volume / total_volume) * 100, 2)

        self._numeric = vmr
        self._value = f"{vmr:.2f}%"
        return self._value

    def get_flag(self):
        if self._numeric is None:
            return 3  # neutre si inconnu

        if self._numeric >= 85:
            return 5  # très concentré
        elif self._numeric >= 70:
            return 4
        elif self._numeric >= 50:
            return 3
        elif self._numeric >= 30:
            return 2
        else:
            return 1  # bien réparti

    def get_label(self):
        return "VMR"

    def get_message(self):
        if self._numeric is None:
            return "La concentration du volume n’a pas pu être calculée."

        if self._numeric >= 85:
            return "L’activité du marché est extrêmement concentrée sur quelques cryptos."
        elif self._numeric >= 70:
            return "Le volume de marché est concentré autour des grandes cryptos."
        elif self._numeric >= 50:
            return "L’activité est modérément concentrée."
        elif self._numeric >= 30:
            return "Le volume est réparti de manière relativement équilibrée."
        else:
            return "Le volume est très bien réparti sur l’ensemble du marché."

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
