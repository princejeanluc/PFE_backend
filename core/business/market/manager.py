from core.business.market.indicators.base import MarketInfoBase
from .indicators.btcdominance import BTCDominanceInfo
from .indicators.declinecount import DeclineCountInfo
from .indicators.pdi import PDIIndicator
from .indicators.topcryptovariation import TopCryptoVariationInfo
from .indicators.upwardtrend import UpwardTrendInfo
from .indicators.vmr import VMRIndicator
from .indicators.volatility import VolatilityInfo
from django.utils.timezone import now , timedelta
from core.models import Crypto, MarketIndicatorSnapshot
from typing import List

class MarketInfoManager:
    def __init__(self):
        self.cryptos = Crypto.objects.all()
        self.indicators  : List[MarketInfoBase]   = [
            VolatilityInfo(self.cryptos),
            TopCryptoVariationInfo(self.cryptos),
            VMRIndicator(self.cryptos),
            UpwardTrendInfo(self.cryptos),
            PDIIndicator(self.cryptos),
            BTCDominanceInfo(self.cryptos),
            DeclineCountInfo(self.cryptos)
            
            # Ajoute ReturnInfo, MarketCapInfo, etc.
        ]

    def get_all_indicators(self):
        results = []
        for indicator_class in self.indicator_classes:
            indicator = indicator_class(self.cryptos)
            label = indicator.get_label()  # Nom lisible pour l’utilisateur
            snapshot = MarketIndicatorSnapshot.objects.filter(name=label).order_by('-updated_at').first()

            if snapshot and snapshot.updated_at > now() - timedelta(minutes=30):
                # Cas : valeur en cache encore fraîche
                results.append({
                    "indicator": snapshot.name,
                    "indicatorValue": snapshot.value,
                    "message": snapshot.message,
                    "colorFlag": snapshot.flag,
                })
            else:
                # Cas : recalcul
                indicator_value = indicator.compute()
                numeric_value = getattr(indicator, "_numeric", None)
                message = indicator.get_message()
                flag = indicator.get_flag()

                MarketIndicatorSnapshot.objects.update_or_create(
                    name=label,
                    defaults={
                        "value": indicator_value,
                        "numeric_value": numeric_value,
                        "message": message,
                        "flag": flag,
                    }
                )

                results.append({
                    "indicator": label,
                    "indicatorValue": indicator_value,
                    "message": message,
                    "colorFlag": flag,
                })

        return results
    
    def save_snapshots(self):
        for indicator in self.indicators:
            indicator.compute()
            if hasattr(indicator, "save_snapshot"):
                indicator.save_snapshot()

