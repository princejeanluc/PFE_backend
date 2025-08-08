from django.db.models import OuterRef, Subquery, F
from heapq import nlargest
from core.models import CryptoInfo, MarketIndicatorSnapshot
from .base import MarketInfoBase

class VMRIndicator(MarketInfoBase):
    WINDOW = None  # instantané
    top_n = 5

    def compute(self):
        # Sous-requête: dernier total_volume par crypto
        latest_vol_sq = (
            CryptoInfo.objects
            .filter(crypto=OuterRef('pk'))
            .order_by('-timestamp')
            .values('total_volume')[:1]
        )

        # On se limite aux cryptos du queryset fourni
        qs = (
            self.crypto_queryset
            .annotate(latest_volume=Subquery(latest_vol_sq), sym=F('symbol'))
            .filter(latest_volume__isnull=False)
        )

        volumes = [c.latest_volume for c in qs if c.latest_volume and c.latest_volume > 0]
        if not volumes:
            self._numeric = None
            self._value = "N/A"
            return self._value

        total_volume = sum(volumes)
        if total_volume <= 0:
            self._numeric = None
            self._value = "N/A"
            return self._value

        top_sum = sum(nlargest(self.top_n, volumes))
        vmr = round((top_sum / total_volume) * 100, 2)

        self._numeric = vmr
        self._value = f"{vmr:.2f}%"
        return self._value

    def get_flag(self):
        # 1 = pire (très concentré), 5 = meilleur (bien réparti)
        v = getattr(self, "_numeric", None)
        if v is None:
            return 3
        if v >= 85:
            return 1
        elif v >= 70:
            return 2
        elif v >= 50:
            return 3
        elif v >= 30:
            return 4
        else:
            return 5

    def get_label(self):
        return "VMR"

    def get_message(self):
        v = getattr(self, "_numeric", None)
        if v is None:
            return "La concentration du volume n’a pas pu être calculée."
        if v >= 85:
            return "Activité extrêmement concentrée sur quelques cryptos."
        elif v >= 70:
            return "Volume concentré autour des grandes cryptos."
        elif v >= 50:
            return "Concentration modérée de l’activité."
        elif v >= 30:
            return "Répartition relativement équilibrée des volumes."
        else:
            return "Volume très bien réparti sur l’ensemble du marché."

    def save_snapshot(self):
        if not hasattr(self, "_value"):
            self.compute()
        MarketIndicatorSnapshot.objects.update_or_create(
            name=self.__class__.__name__,
            defaults={
                "value": self._value,
                "numeric_value": self._numeric,
                "flag": self.get_flag(),
                "message": self.get_message(),
            }
        )
