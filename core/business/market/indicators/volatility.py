from core.models import CryptoInfo, MarketIndicatorSnapshot
from .base import MarketInfoBase
from datetime import timedelta
from django.utils.timezone import now
import numpy as np
import math
from collections import defaultdict

class VolatilityInfo(MarketInfoBase):
    """
    Volatilité sur les dernières 24h (écart-type des rendements log),
    annualisé sur base 24h.
    """
    def __init__(self, crypto_queryset):
        super().__init__(crypto_queryset)
        self._numeric = None
        self._value = None

    def compute(self):
        time_threshold = now() - timedelta(days=1)

        # Récupérer toutes les observations récentes
        recent_infos = (
            CryptoInfo.objects
            .filter(timestamp__gte=time_threshold, current_price__isnull=False)
            .filter(crypto__in=self.crypto_queryset)
            .order_by('crypto', 'timestamp')
        )

        prices_by_crypto = defaultdict(list)
        for info in recent_infos:
            prices_by_crypto[info.crypto.symbol].append(info.current_price)

        returns = []
        for price_list in prices_by_crypto.values():
            if len(price_list) < 2:
                continue
            for p0, p1 in zip(price_list[:-1], price_list[1:]):
                if p0 > 0 and p1 > 0:
                    returns.append(math.log(p1 / p0))

        if not returns:
            self._numeric = None
            self._value = "N/A"
            return self._value

        vol = np.std(returns) * np.sqrt(24)  # "annualisation" sur 24h
        self._numeric = vol
        self._value = f"{vol:.2%}"
        return self._value

    def get_flag(self):
        if self._numeric is None:
            return 3  # neutre
        if self._numeric > 0.10:
            return 1  # élevé
        elif self._numeric > 0.05:
            return 2  # modéré
        return 3  # faible

    def get_label(self):
        return "Volatilité"

    def get_message(self):
        if self._numeric is None:
            return "Volatilité non calculable (pas assez de données)."
        if self._numeric > 0.10:
            return "Volatilité élevée : prudence recommandée."
        elif self._numeric > 0.05:
            return "Volatilité modérée : surveillez le marché."
        return "Volatilité faible : marché stable."

    def save_snapshot(self):
        if self._value is None:
            self.compute()
        MarketIndicatorSnapshot.objects.update_or_create(
            name=self.get_label(),
            defaults={
                "value": self._value,
                "numeric_value": self._numeric,
                "flag": self.get_flag(),
                "message": self.get_message(),
            }
        )
