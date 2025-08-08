from .base import MarketInfoBase
from core.models import CryptoInfo, MarketIndicatorSnapshot
from datetime import timedelta
from django.utils.timezone import now
import numpy as np
from collections import defaultdict

class AverageTopVolatilityInfo(MarketInfoBase):
    WINDOW_DAYS = 30  # Fenêtre optimale pour cet indicateur
    RECENT_HOURS = 12  # Fenêtre pour déterminer le top 10

    def compute(self):
        now_time = now()
        time_threshold = now_time - timedelta(days=self.WINDOW_DAYS)
        recent_cutoff = now_time - timedelta(hours=self.RECENT_HOURS)

        # Étape 1 : récupérer les dernières market_cap par crypto (sur fenêtre récente)
        latest_infos = (
            CryptoInfo.objects
            .filter(timestamp__gte=recent_cutoff, market_cap__isnull=False)
            .order_by('crypto', '-timestamp')
            .distinct('crypto')  # PostgreSQL requis
        )

        latest_market_cap = {}
        for info in latest_infos:
            symbol = info.crypto.symbol
            if symbol not in latest_market_cap:
                latest_market_cap[symbol] = (info.market_cap, info.crypto)

        # Top 10 par market_cap
        top_cryptos = sorted(latest_market_cap.items(), key=lambda x: x[1][0], reverse=True)[:10]
        top_symbols = {crypto.symbol for _, (_, crypto) in top_cryptos}

        # Étape 2 : récupérer les prix sur la fenêtre
        recent_infos = (
            CryptoInfo.objects
            .filter(timestamp__gte=time_threshold, current_price__isnull=False)
            .filter(crypto__symbol__in=top_symbols)
            .order_by('crypto', 'timestamp')
        )

        # Étape 3 : agréger par crypto
        prices_by_crypto = defaultdict(list)
        for info in recent_infos:
            prices_by_crypto[info.crypto.symbol].append(info.current_price)

        # Étape 4 : calcul des volatilités
        volatilities = []
        for prices in prices_by_crypto.values():
            if len(prices) >= 2:
                log_returns = np.diff(np.log(prices))
                if len(log_returns):
                    vol = np.std(log_returns) * np.sqrt(24)
                    volatilities.append(vol)

        # Résultat final
        if not volatilities:
            self._value = "N/A"
            self._numeric = None
        else:
            avg_vol = float(np.mean(volatilities))
            self._value = f"{avg_vol:.2%}"
            self._numeric = avg_vol

        return self._value

    def get_flag(self):
        if self._numeric is None:
            return 3  # neutre

        v = self._numeric
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
        return "Volatilité moyenne du Top 10"

    def get_message(self):
        if self._numeric is None:
            return "Volatilité non disponible pour le top crypto."

        v = self._numeric
        if v > 0.10:
            return "Instabilité extrême sur les cryptos majeures."
        elif v > 0.06:
            return "Volatilité élevée sur le top 10."
        elif v > 0.03:
            return "Marché modérément instable parmi les leaders."
        elif v > 0.01:
            return "Stabilité relative dans le top crypto."
        else:
            return "Très faible volatilité parmi les cryptos dominantes."

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
