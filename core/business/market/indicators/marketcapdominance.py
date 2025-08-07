

from core.business.market.indicators.base import MarketInfoBase


class MarketCapDominanceIndex(MarketInfoBase):
    def __init__(self, crypto_queryset, top_n=5):
        super().__init__(crypto_queryset)
        self.top_n = top_n

    def compute(self):
        # Extraire les market_caps valides
        market_caps = [
            crypto.market_cap
            for crypto in self.crypto_queryset
            if crypto.market_cap is not None and crypto.market_cap > 0
        ]

        if len(market_caps) == 0:
            return "N/A"

        # Tri décroissant
        sorted_caps = sorted(market_caps, reverse=True)

        top_n = min(self.top_n, len(sorted_caps))
        top_sum = sum(sorted_caps[:top_n])
        total_sum = sum(sorted_caps)

        dominance_ratio = top_sum / total_sum

        return round(dominance_ratio, 4)  # Ex: 0.7832 = 78.32%

    def get_label(self):
        return f"Top {self.top_n} Market Cap Dominance"

    def get_flag(self):
        ratio = self.compute()
        if ratio == "N/A":
            return "gray"

        if ratio > 0.75:
            return "orange"  # Très concentré
        elif ratio > 0.6:
            return "yellow"  # Moyennement concentré
        else:
            return "green"   # Diversifié

    def get_message(self):
        ratio = self.compute()
        if ratio == "N/A":
            return "Impossible de calculer la dominance du marché (pas de données valides)."
        pct = round(ratio * 100, 2)
        return f"Les {self.top_n} premières cryptomonnaies représentent {pct}% de la capitalisation totale du marché."
