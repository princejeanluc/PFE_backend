from .base import MarketInfoBase
from core.models import CryptoInfo, MarketIndicatorSnapshot
from django.utils.timezone import now, timedelta
import numpy as np
import pandas as pd

class TopCryptoVariationInfo(MarketInfoBase):
    def compute(self):
        end_time = now()
        start_time = end_time - timedelta(hours=24)

        # Récupère les dernières infos pour chaque crypto
        all_infos = (
            CryptoInfo.objects
            .filter(timestamp__lte=end_time)
            .order_by('-timestamp')
        )

        latest_info_per_crypto = {}
        for info in all_infos:
            symbol = info.crypto.symbol
            if symbol not in latest_info_per_crypto and info.market_cap is not None:
                latest_info_per_crypto[symbol] = info

        # Top 10 par capitalisation
        top_cryptos = sorted(
            latest_info_per_crypto.values(),
            key=lambda x: x.market_cap,
            reverse=True
        )[:10]

        variations = []
        for info in top_cryptos:
            crypto = info.crypto
            past_info = (
                CryptoInfo.objects
                .filter(crypto=crypto, timestamp__lte=start_time)
                .order_by('-timestamp')
                .first()
            )
            if past_info and past_info.current_price and info.current_price:
                past_price = past_info.current_price
                current_price = info.current_price
                if past_price > 0:
                    variation = (current_price - past_price) / past_price
                    variations.append(variation)

        if not variations:
            self._value = "N/A"
            self._numeric = None
        else:
            avg_var = np.mean(variations)
            self._value = f"{avg_var:.2%}"        # ex : "4.32%"
            self._numeric = round(avg_var, 4)     # ex : 0.0432
            self._avg_variation = avg_var

        return self._value

    def get_flag(self):
        if hasattr(self, "_avg_variation"):
            var = self._avg_variation
            if var >= 0.10:
                return 5
            elif var >= 0.05:
                return 4
            elif var > -0.05:
                return 3
            elif var > -0.10:
                return 2
            else:
                return 1
        return 3

    def get_label(self):
        return "Variation 24h (Top 10)"

    def get_message(self):
        if hasattr(self, "_avg_variation"):
            var = self._avg_variation
            if var >= 0.10:
                return "Croissance forte du top 10 crypto : tendance haussière nette."
            elif var >= 0.05:
                return "Croissance modérée du top 10 crypto : tendance haussière."
            elif var > -0.05:
                return "Variation modérée du top 10 crypto sur 24h."
            elif var > -0.10:
                return "Baisse modérée du top 10 crypto."
            else:
                return "Baisse marquée du top 10 crypto."
        return "Variation stable des cryptos principales."

    def save_snapshot(self):
        MarketIndicatorSnapshot.objects.create(
            name=self.get_label(),
            value=self._value,
            numeric_value=self._numeric,
            flag=self.get_flag(),
            message=self.get_message()
        )
