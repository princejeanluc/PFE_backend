from django.db.models import OuterRef, Subquery
from core.models import CryptoInfo, MarketIndicatorSnapshot
from .base import MarketInfoBase

class MCDIIndicator(MarketInfoBase):
    """
    Market Cap Dominance Index (Top N dominance en %).
    Mesure la part de capitalisation captée par les N plus grosses cryptos.
    Plus c'est élevé, plus le marché est concentré (donc 'pire' pour la diversification).
    """
    def __init__(self, crypto_queryset, top_n=5):
        super().__init__(crypto_queryset)
        self.top_n = top_n
        self._numeric = None
        self._value = None

    def compute(self):
        # Dernière market_cap par crypto (pas d'hypothèse d'un timestamp commun)
        latest_cap_sq = (CryptoInfo.objects
                         .filter(crypto=OuterRef('pk'))
                         .order_by('-timestamp')
                         .values('market_cap')[:1])

        # On annote le queryset passé au constructeur (meilleur contrôle que Crypto.objects.all())
        qs = self.crypto_queryset.annotate(latest_market_cap=Subquery(latest_cap_sq))\
                                 .filter(latest_market_cap__isnull=False)

        caps = [c.latest_market_cap for c in qs if c.latest_market_cap and c.latest_market_cap > 0]
        if not caps:
            self._numeric = None
            self._value = "N/A"
            return self._value

        caps.sort(reverse=True)
        top_sum = sum(caps[:min(self.top_n, len(caps))])
        total_sum = sum(caps)
        if total_sum <= 0:
            self._numeric = None
            self._value = "N/A"
            return self._value

        dominance_pct = round((top_sum / total_sum) * 100, 2)  # en %
        self._numeric = dominance_pct
        self._value = f"{dominance_pct:.2f}%"
        return self._value

    def get_flag(self):
        # 1 = pire (très concentré), 5 = meilleur (diversifié)
        if self._numeric is None:
            return 3  # neutre par défaut si on ne sait pas
        v = self._numeric
        if v >= 80:
            return 1
        elif v >= 65:
            return 2
        elif v >= 50:
            return 3
        elif v >= 35:
            return 4
        else:
            return 5

    def get_label(self):
        return f"MCDI (Top {self.top_n})"

    def get_message(self):
        if self._numeric is None:
            return "La concentration du marché n'a pas pu être évaluée."
        v = self._numeric
        if v >= 80:
            return "Marché très concentré : quelques cryptomonnaies dominent largement."
        elif v >= 65:
            return "Marché concentré : poids important des grandes capitalisations."
        elif v >= 50:
            return "Concentration modérée du marché."
        elif v >= 35:
            return "Concentration faible : capitalisation relativement répartie."
        else:
            return "Marché bien diversifié : faible dominance du Top."

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
