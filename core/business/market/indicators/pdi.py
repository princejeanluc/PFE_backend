from django.db.models import OuterRef, Subquery
from core.models import CryptoInfo, MarketIndicatorSnapshot
from .base import MarketInfoBase

class PDIIndicator(MarketInfoBase):
    """
    Price Dispersion Indicator : mesure l'écart-type relatif des prix
    entre cryptos, en pourcentage de la moyenne.
    """
    WINDOW = None  # Instantané (dernière valeur par crypto)

    def __init__(self, crypto_queryset):
        super().__init__(crypto_queryset)
        self._numeric = None  # valeur numérique (%)
        self._value = None    # version texte (%)

    def compute(self):
        # Dernier prix pour chaque crypto (pas d’hypothèse d'un timestamp commun)
        latest_price_sq = (CryptoInfo.objects
                           .filter(crypto=OuterRef('pk'))
                           .order_by('-timestamp')
                           .values('current_price')[:1])

        qs = (self.crypto_queryset
              .annotate(latest_price=Subquery(latest_price_sq))
              .filter(latest_price__isnull=False, latest_price__gt=0))

        prices = [c.latest_price for c in qs]
        if not prices:
            self._numeric = None
            self._value = "N/A"
            return self._value

        mean_price = sum(prices) / len(prices)
        variance = sum((p - mean_price) ** 2 for p in prices) / len(prices)
        std_dev = variance ** 0.5

        dispersion_pct = round((std_dev / mean_price) * 100, 2)
        self._numeric = dispersion_pct
        self._value = f"{dispersion_pct:.2f}%"
        return self._value

    def get_flag(self):
        if self._numeric is None:
            return 3  # neutre par défaut
        v = self._numeric
        if v >= 100:
            return 1  # dispersion extrême
        elif v >= 75:
            return 2  # forte dispersion
        elif v >= 50:
            return 3  # dispersion modérée
        elif v >= 25:
            return 4  # faible dispersion
        else:
            return 5  # très faible dispersion

    def get_label(self):
        return "PDI"

    def get_message(self):
        if self._numeric is None:
            return "La dispersion des prix n'a pas pu être calculée."
        v = self._numeric
        if v >= 100:
            return "Marché extrêmement dispersé : prix très éloignés les uns des autres."
        elif v >= 75:
            return "Marché très dispersé : grandes différences de prix."
        elif v >= 50:
            return "Dispersion modérée des prix."
        elif v >= 25:
            return "Marché relativement homogène."
        else:
            return "Marché très homogène : prix proches entre cryptos."

    def get_message_help(self):
        return (
            "Le PDI (Price Dispersion Indicator) mesure à quel point les prix des cryptomonnaies "
            "sont dispersés par rapport à leur moyenne. Une forte dispersion peut indiquer des écarts "
            "significatifs entre cryptos, tandis qu'une faible dispersion reflète un marché plus homogène."
        )

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
