from core.models import CryptoInfo
from .base import MarketInfoBase

class VMRIndicator(MarketInfoBase):
    def __init__(self, crypto_queryset):
        super().__init__(crypto_queryset)
        self.top_n = 5
        self.result = None

    def compute(self):
        latest_date = CryptoInfo.objects.latest("timestamp").timestamp.date()
        data = CryptoInfo.objects.filter(timestamp__date=latest_date)

        total_volume = sum(item.volume for item in data if item.volume)
        top_cryptos = sorted(data, key=lambda x: x.volume or 0, reverse=True)[:self.top_n]
        top_volume = sum(item.volume for item in top_cryptos if item.volume)

        if total_volume == 0:
            self.result = None
        else:
            self.result = round((top_volume / total_volume) * 100, 2)

        return self.result

    def get_flag(self):
        if self.result is None:
            return 0
        if self.result >= 85:
            return 5  # Très concentré
        elif self.result >= 70:
            return 4
        elif self.result >= 50:
            return 3
        elif self.result >= 30:
            return 2
        else:
            return 1  # Bien réparti

    def get_label(self):
        return "VMR"

    def get_message(self):
        if self.result is None:
            return "La concentration du volume n’a pas pu être calculée."

        if self.result >= 85:
            return "L’activité du marché est extrêmement concentrée sur quelques cryptos."
        elif self.result >= 70:
            return "Le volume de marché est concentré autour des grandes cryptos."
        elif self.result >= 50:
            return "L’activité est modérément concentrée."
        elif self.result >= 30:
            return "Le volume est réparti de manière relativement équilibrée."
        else:
            return "Le volume est très bien réparti sur l’ensemble du marché."
