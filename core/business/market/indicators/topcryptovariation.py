from .base import MarketInfoBase
from core.models import CryptoInfo
from django.utils.timezone import now, timedelta
import numpy as np

class TopCryptoVariationInfo(MarketInfoBase):
    def compute(self):
        end_time = now()
        start_time = end_time - timedelta(hours=24)

        # Récupère les dernières infos
        all_infos = (
            CryptoInfo.objects
            .filter(timestamp__lte=end_time)
            .order_by('-timestamp')
        )

        # Dictionnaire pour ne garder qu’une seule info par crypto
        latest_info_per_crypto = {}
        for info in all_infos:
            symbol = info.crypto.symbol
            if symbol not in latest_info_per_crypto and info.market_cap is not None:
                latest_info_per_crypto[symbol] = info

        # Trie par market cap et prend le top 10
        top_cryptos = sorted(
            latest_info_per_crypto.values(),
            key=lambda x: x.market_cap,
            reverse=True
        )[:10]

        variations = []
        for info in top_cryptos:
            crypto = info.crypto
            # Info 24h en arrière
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
            return "N/A"

        avg_variation = np.mean(variations)
        self._avg_variation = avg_variation
        return f"{avg_variation:.2%}"

    def get_flag(self):
        if hasattr(self, "_avg_variation"):
            var = abs(self._avg_variation)
            if var >= 0.10:
                return 1
            elif var >= 0.05:
                return 2
        return 3

    def get_label(self):
        return "Variation 24h (Top 10)"

    def get_message(self):
        if hasattr(self, "_avg_variation"):
            if self._avg_variation > 0.05:
                return "Croissance moyenne du top 10 crypto : tendance haussière marquée."
            elif self._avg_variation < -0.05:
                return "Baisse moyenne du top 10 crypto : tendance baissière notable."
            else:
                return "Variation modérée du top 10 crypto sur 24h."
        return "Variation stable des cryptos principales."
