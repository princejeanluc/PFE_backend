from django.db.models import OuterRef, Subquery, F
from core.models import Crypto, CryptoInfo, MarketIndicatorSnapshot
from .base import MarketInfoBase
from heapq import nlargest

class MarketCapDominanceIndex(MarketInfoBase):
    # Fenêtre: instantané (photo du marché)
    WINDOW = None

    def __init__(self, crypto_queryset, top_n=5):
        super().__init__(crypto_queryset)
        self.top_n = top_n
        self._numeric = None
        self._value = None

    def compute(self):
        # Sous-requête: dernière market_cap par crypto
        latest_info_sq = (
            CryptoInfo.objects
            .filter(crypto=OuterRef('pk'))
            .order_by('-timestamp')
            .values('market_cap')[:1]
        )

        # Annoter le queryset fourni (on ne scanne pas tout le marché)
        qs = (
            self.crypto_queryset
            .annotate(
                latest_market_cap=Subquery(latest_info_sq),
                sym=F('symbol')
            )
            .filter(latest_market_cap__isnull=False)
        )

        # Extraire uniquement les caps valides (>0)
        caps = [c.latest_market_cap for c in qs if c.latest_market_cap and c.latest_market_cap > 0]
        if not caps:
            self._numeric = None
            self._value = "N/A"
            return self._value

        # Top-N sans trier toute la liste
        top_vals = nlargest(min(self.top_n, len(caps)), caps)
        top_sum = sum(top_vals)
        total_sum = sum(caps)

        if total_sum <= 0:
            self._numeric = None
            self._value = "N/A"
            return self._value

        dominance_pct = round((top_sum / total_sum) * 100, 2)  # %
        self._numeric = dominance_pct
        self._value = f"{dominance_pct:.2f}%"
        return self._value

    def get_flag(self):
        # 1 = pire (très concentré), 5 = meilleur (diversifié)
        if self._numeric is None:
            return 3
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
            return "Impossible d’évaluer la dominance (données indisponibles)."
        v = self._numeric
        if v >= 80:
            return "Marché très concentré : quelques cryptos captent l’essentiel de la capitalisation."
        elif v >= 65:
            return "Marché concentré : poids important des grandes capitalisations."
        elif v >= 50:
            return "Concentration modérée du marché."
        elif v >= 35:
            return "Concentration faible : capitalisation relativement répartie."
        else:
            return "Marché bien diversifié : faible concentration du Top."

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
