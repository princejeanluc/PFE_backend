from .base import MarketInfoBase
from core.models import CryptoInfo
from datetime import timedelta
from django.utils.timezone import now
import numpy as np
import math

class VolatilityInfo(MarketInfoBase):
    def compute(self):
        time_threshold = now() - timedelta(days=1)
        recent_infos = (
            CryptoInfo.objects.filter(timestamp__gte=time_threshold, current_price__isnull=False)
            .order_by('crypto', 'timestamp')
        )

        returns = []
        prices_by_crypto = {}

        for info in recent_infos:
            symbol = info.crypto.symbol
            price = info.current_price
            if symbol not in prices_by_crypto:
                prices_by_crypto[symbol] = []
            prices_by_crypto[symbol].append(price)

        for price_list in prices_by_crypto.values():
            if len(price_list) < 2:
                continue
            for i in range(1, len(price_list)):
                p1, p0 = price_list[i], price_list[i-1]
                if p0 > 0 and p1 > 0:
                    log_return = math.log(p1 / p0)
                    returns.append(log_return)

        if not returns:
            return "N/A"

        vol = np.std(returns) * np.sqrt(24)  # annualise-like scale for 24h
        self._vol = vol
        return f"{vol:.2%}"  # format en pourcentage

    def get_flag(self):
        if hasattr(self, "_vol"):
            if self._vol > 0.10:
                return 1  # élevé
            elif self._vol > 0.05:
                return 2  # modéré
        return 3  # faible

    def get_label(self):
        return "Volatilité"

    def get_message(self):
        if hasattr(self, "_vol"):
            if self._vol > 0.10:
                return "Volatilité élevée : prudence recommandée."
            elif self._vol > 0.05:
                return "Volatilité modérée : surveillez le marché."
        return "Volatilité faible : marché stable."
