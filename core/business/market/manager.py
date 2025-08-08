# core/business/market/manager.py
from typing import List, Type, Dict, Any, Optional
from django.db import transaction
from django.utils.timezone import now, timedelta
from django.db.models import QuerySet

from core.business.market.indicators.btcdominance import BTCDominanceInfo
from core.business.market.indicators.topcryptovariation import TopCryptoVariationInfo
from core.business.market.indicators.upwardtrend import UpwardTrendInfo
from core.business.market.indicators.vmr import VMRIndicator
from core.models import Crypto, MarketIndicatorSnapshot
from .indicators.base import MarketInfoBase

# Tes imports d'indicateurs
from .indicators.volatility import VolatilityInfo
from .indicators.pdi import PDIIndicator
from .indicators.declinecount import DeclineCountInfo
# from .indicators.topcryptovariation import TopCryptoVariationInfo
# from .indicators.vmr import VMRIndicator
# from .indicators.upwardtrend import UpwardTrendInfo
# from .indicators.btcdominance import BTCDominanceInfo

class MarketInfoManager:
    """
    - Sert des snapshots si récents (TTL par indicateur)
    - Sinon calcule, met à jour le snapshot, et retourne.
    - Fournit aussi une méthode save_snapshots() (cron/Azure) sans bug.
    """

    # Déclare ici tes indicateurs actifs
    indicator_classes: List[Type[MarketInfoBase]] = [
        VolatilityInfo,
        PDIIndicator,
        DeclineCountInfo,
        TopCryptoVariationInfo,
        VMRIndicator,
        UpwardTrendInfo,
        BTCDominanceInfo,
    ]

    # TTL par indicateur (minutes). Ajuste selon coût / nature du signal.
    freshness_minutes: Dict[str, int] = {
        "Volatilité": 60,            # moyen/coûteux → 1h
        "PDI": 60 * 24,              # structurel → 24h
        "Chutes > 10%": 30,          # réactif → 30 min
        "Variation 24h (Top 10)": 60,
        "VMR": 30,
        "Cryptos en hausse": 30,
        "Dominance BTC": 60 * 6,   # 6h
    }

    def __init__(self, cryptos: Optional[QuerySet] = None):
        # Limite le queryset aux champs utiles pour réduire le poids en mémoire
        self.cryptos = (cryptos or Crypto.objects.all()).only("id", "symbol")

    def _ttl_for(self, label: str) -> timedelta:
        minutes = self.freshness_minutes.get(label, 30)  # défaut 30 min
        return timedelta(minutes=minutes)

    def _compute_and_snapshot(self, indicator: MarketInfoBase) -> Dict[str, Any]:
        label = indicator.get_label()
        try:
            value = indicator.compute()
            numeric = getattr(indicator, "_numeric", None)
            flag = indicator.get_flag()
            message = indicator.get_message()
        except Exception as e:
            # Sécurité: si compute plante, on renvoie un snapshot “erreur”
            value, numeric, flag = "N/A", None, 3
            message = f"Erreur lors du calcul ({label})."
        # Écrit en base de façon atomique
        with transaction.atomic():
            MarketIndicatorSnapshot.objects.update_or_create(
                name=label,
                defaults={
                    "value": value,
                    "numeric_value": numeric,
                    "flag": flag,
                    "message": message,
                }
            )
        return {
            "indicator": label,
            "indicatorValue": value,
            "message": message,
            "colorFlag": flag,
        }

    def get_all_indicators(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        for cls in self.indicator_classes:
            indicator = cls(self.cryptos)
            label = indicator.get_label()

            snapshot = (
                MarketIndicatorSnapshot.objects
                .filter(name=label)
                .order_by("-updated_at")
                .first()
            )

            ttl = self._ttl_for(label)
            if snapshot and snapshot.updated_at and snapshot.updated_at > now() - ttl:
                # Données fraîches → on serre le cache
                results.append({
                    "indicator": snapshot.name,
                    "indicatorValue": snapshot.value,
                    "message": snapshot.message,
                    "colorFlag": snapshot.flag,
                })
            else:
                # Recalcule et met à jour snapshot
                results.append(self._compute_and_snapshot(indicator))

        return results

    def save_snapshots(self) -> None:
        """
        À appeler depuis un cron / Azure Function.
        Calcul séquentiel de chaque indicateur et écriture snapshot.
        (Correction: on n'utilise plus self.indicators, qui n’existe pas.)
        """
        for cls in self.indicator_classes:
            indicator = cls(self.cryptos)
            self._compute_and_snapshot(indicator)
