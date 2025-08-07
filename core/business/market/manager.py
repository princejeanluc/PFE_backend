from .indicators.btcdominance import BTCDominanceInfo
from .indicators.declinecount import DeclineCountInfo
from .indicators.pdi import PDIIndicator
from .indicators.topcryptovariation import TopCryptoVariationInfo
from .indicators.upwardtrend import UpwardTrendInfo
from .indicators.vmr import VMRIndicator
from .indicators.volatility import VolatilityInfo

from core.models import Crypto

class MarketInfoManager:
    def __init__(self):
        self.cryptos = Crypto.objects.all()
        self.indicators = [
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
        return [indicator.get_info() for indicator in self.indicators]
