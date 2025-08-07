from core.models import CryptoInfo
from .base import MarketInfoBase
from datetime import datetime

class PDIIndicator(MarketInfoBase):
    def __init__(self, crypto_queryset):
        super().__init__(crypto_queryset)
        self.result = None

    def compute(self):
        latest_date = CryptoInfo.objects.latest("timestamp").timestamp.date()
        data = CryptoInfo.objects.filter(timestamp__date=latest_date)

        prices = [item.current_price for item in data if item.current_price and item.current_price > 0]

        if not prices:
            self.result = None
        else:
            mean_price = sum(prices) / len(prices)
            squared_diff = [(p - mean_price) ** 2 for p in prices]
            variance = sum(squared_diff) / len(prices)
            std_dev = variance ** 0.5

            dispersion = (std_dev / mean_price) * 100  # exprimé en %
            self.result = round(dispersion, 2)

        return self.result

    def get_flag(self):
        if self.result is None:
            return 0
        if self.result >= 100:
            return 1  # dispersion extrême
        elif self.result >= 75:
            return 2  # forte dispersion
        elif self.result >= 50:
            return 3  # dispersion modérée
        elif self.result >= 25:
            return 4  # faible dispersion
        else:
            return 5  # très faible dispersion, homogène

    def get_label(self):
        return "PDI"

    def get_message(self):
        if self.result is None:
            return "La dispersion des prix n'a pas pu être calculée."

        if self.result >= 100:
            return "Le marché est extrêmement dispersé, les cryptos sont très éloignées les unes des autres."
        elif self.result >= 75:
            return "Le marché est très dispersé, avec de grandes différences de prix."
        elif self.result >= 50:
            return "Le marché montre une dispersion modérée des prix."
        elif self.result >= 25:
            return "Le marché est relativement homogène en termes de prix."
        else:
            return "Les cryptos ont des prix très proches les uns des autres, marché très homogène."

    def get_message_help(self):
        return (
            "Le PDI (Price Dispersion Indicator) mesure à quel point les prix des cryptomonnaies "
            "sont dispersés par rapport à leur moyenne. Une forte dispersion peut indiquer des écarts "
            "significatifs entre cryptos, tandis qu'une faible dispersion reflète un marché plus homogène."
        )
