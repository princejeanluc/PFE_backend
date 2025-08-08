from core.models import CryptoInfo, MarketIndicatorSnapshot
from .base import MarketInfoBase
from django.utils.timezone import now, timedelta
import numpy as np
import math

class VolatilityInfo(MarketInfoBase):
    """
    Volatilité sur les dernières 24h (écart-type des rendements log),
    annualisé-like sur base 24 (sqrt(24)).
    """
    WINDOW_HOURS = 24  # fenêtre explicite

    def __init__(self, crypto_queryset):
        super().__init__(crypto_queryset)
        self._numeric = None
        self._value = None

    def compute(self):
        time_threshold = now() - timedelta(hours=self.WINDOW_HOURS)

        # Requête allégée: on ne ramène que (crypto_id, price) triés
        rows = (
            CryptoInfo.objects
            .filter(
                timestamp__gte=time_threshold,
                current_price__isnull=False,
                crypto__in=self.crypto_queryset,
            )
            .order_by('crypto_id', 'timestamp')
            .values_list('crypto_id', 'current_price')
        )

        returns = []
        last_price = None
        last_crypto = None

        for crypto_id, price in rows.iterator(chunk_size=2000):
            if last_crypto != crypto_id:
                # on change de crypto
                last_crypto = crypto_id
                last_price = price
                continue
            # même crypto: calcul du log-return si données valides
            if last_price and price and last_price > 0 and price > 0:
                returns.append(math.log(price / last_price))
            last_price = price

        if not returns:
            self._numeric = None
            self._value = "N/A"
            return self._value

        vol = float(np.std(returns) * np.sqrt(24))
        self._numeric = vol
        self._value = f"{vol:.2%}"
        return self._value

    def get_flag(self):
        # 1 = pire (volatilité très élevée), 5 = meilleur (très faible)
        v = getattr(self, "_numeric", None)
        if v is None:
            return 3  # neutre si inconnu
        if v > 0.10:
            return 1
        elif v > 0.06:
            return 2
        elif v > 0.03:
            return 3
        elif v > 0.01:
            return 4
        else:
            return 5

    def get_label(self):
        return "Volatilité"

    def get_message(self):
        v = getattr(self, "_numeric", None)
        if v is None:
            return "Volatilité non calculable (pas assez de données)."
        if v > 0.10:
            return "Volatilité élevée : prudence recommandée."
        elif v > 0.06:
            return "Volatilité modérée : surveillez le marché."
        elif v > 0.03:
            return "Volatilité contenue : conditions assez stables."
        elif v > 0.01:
            return "Volatilité faible : marché plutôt stable."
        else:
            return "Volatilité très faible : marché calme."

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
