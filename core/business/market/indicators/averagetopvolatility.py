from .base import MarketInfoBase
from core.models import CryptoInfo
from datetime import timedelta
from django.utils.timezone import now
import numpy as np
import math
from collections import defaultdict

class AverageTopVolatilityInfo(MarketInfoBase):
    def compute(self):
        time_threshold = now() - timedelta(days=30)
        
        # Récupère uniquement les dernières capitalisations disponibles pour chaque crypto
        latest_infos = (
            CryptoInfo.objects
            .filter(timestamp__gte=now() - timedelta(hours=12), market_cap__isnull=False)
            .order_by('crypto', '-timestamp')
        )

        latest_market_cap = {}
        for info in latest_infos:
            symbol = info.crypto.symbol
            if symbol not in latest_market_cap:
                latest_market_cap[symbol] = (info.market_cap, info.crypto)

        # Trie les cryptos par capitalisation décroissante
        top_cryptos = sorted(latest_market_cap.items(), key=lambda x: x[1][0], reverse=True)[:10]
        top_symbols = {crypto.symbol for _, (_, crypto) in top_cryptos}

        # Récupère les prix sur les 30 derniers jours
        recent_infos = (
            CryptoInfo.objects
            .filter(timestamp__gte=time_threshold, current_price__isnull=False)
            .filter(crypto__symbol__in=top_symbols)
            .order_by('crypto', 'timestamp')
        )

        prices_by_crypto = defaultdict(list)
        for info in recent_infos:
            prices_by_crypto[info.crypto.symbol].append(info.current_price)

        # Calcule la volatilité pour chaque crypto
        volatilities = []
        for prices in prices_by_crypto.values():
            if len(prices) < 2:
                continue
            returns = [
                math.log(p1 / p0) for p0, p1 in zip(prices[:-1], prices[1:])
                if p0 > 0 and p1 > 0
            ]
            if returns:
                vol = np.std(returns) * np.sqrt(24)
                volatilities.append(vol)

        if not volatilities:
            return "N/A"

        self._avg_vol = np.mean(volatilities)
        return f"{self._avg_vol:.2%}"

    def get_flag(self):
        if hasattr(self, "_avg_vol"):
            if self._avg_vol > 0.10:
                return 1  # trop élevé
            elif self._avg_vol > 0.06:
                return 2  # élevé
            elif self._avg_vol > 0.03:
                return 3  # modéré
            elif self._avg_vol > 0.01:
                return 4  # faible
        return 5  # très faible

    def get_label(self):
        return "Volatilité moyenne du Top 10"

    def get_message(self):
        if hasattr(self, "_avg_vol"):
            if self._avg_vol > 0.10:
                return "Instabilité extrême sur les cryptos majeures."
            elif self._avg_vol > 0.06:
                return "Volatilité élevée sur le top 10."
            elif self._avg_vol > 0.03:
                return "Marché modérément instable parmi les leaders."
            elif self._avg_vol > 0.01:
                return "Stabilité relative dans le top crypto."
        return "Très faible volatilité parmi les cryptos dominantes."
