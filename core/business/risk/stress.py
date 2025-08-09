# core/business/stress.py
from __future__ import annotations
from typing import Dict, List, Tuple
from datetime import timedelta
import pandas as pd

from django.db.models import OuterRef, Subquery
from django.utils.timezone import now

from core.models import (
    StressScenario,
    Portfolio,
    PortfolioPerformance,
    Holding,
    Crypto,
    CryptoInfo,
)

# ---------- Utils prix courants ----------

def get_latest_prices_for_cryptos(cryptos: List[Crypto], lookback_days: int = 7) -> Dict[str, float]:
    """
    Retourne le dernier current_price disponible pour chaque crypto (par symbol),
    en cherchant sur 'lookback_days' jours.
    """
    since = now() - timedelta(days=lookback_days)
    prices: Dict[str, float] = {}

    # Subquery pour récupérer l'ID du dernier CryptoInfo par crypto depuis 'since'
    latest_info_subq = (
        CryptoInfo.objects
        .filter(crypto=OuterRef('pk'), timestamp__gte=since, current_price__isnull=False)
        .order_by('-timestamp')
        .values('id')[:1]
    )

    # Récupérer les IDs en une passe
    crypto_ids = (
        Crypto.objects
        .filter(id__in=[c.id for c in cryptos])
        .annotate(latest_info_id=Subquery(latest_info_subq))
        .values_list('id', 'latest_info_id')
    )

    info_ids = [info_id for _, info_id in crypto_ids if info_id is not None]
    info_map = {
        ci.crypto_id: float(ci.current_price)
        for ci in CryptoInfo.objects.filter(id__in=info_ids)
    }

    for c in cryptos:
        if c.id in info_map:
            prices[c.symbol] = info_map[c.id]

    return prices

# ---------- Historical replay (agrégation médiane 1H) ----------

def historical_returns_for_symbols(
    symbols: List[str],
    start,
    end,
    proxy_symbol: str | None = None,
    notes: List[str] | None = None,
) -> Dict[str, float]:
    """
    Rendement cumulé par symbole dans [start, end], via CryptoInfo (médiane 1H).
    Si une série est indisponible -> fallback sur proxy_symbol si fourni, sinon 0.0.
    """
    if notes is None:
        notes = []

    # map symbol -> Crypto
    cryptos = {c.symbol: c for c in Crypto.objects.filter(symbol__in=symbols)}
    returns: Dict[str, float] = {}

    # calc proxy à la volée si demandé
    proxy_ret: float | None = None
    if proxy_symbol:
        proxy_crypto = cryptos.get(proxy_symbol) or Crypto.objects.filter(symbol=proxy_symbol).first()
        if proxy_crypto:
            proxy_ret = _cumulative_return_for_crypto(proxy_crypto, start, end)
            if proxy_ret is None:
                notes.append(f"Proxy {proxy_symbol}: pas de données sur la fenêtre.")
                proxy_ret = 0.0
        else:
            notes.append(f"Proxy {proxy_symbol}: crypto introuvable.")
            proxy_ret = 0.0

    for sym in symbols:
        c = cryptos.get(sym)
        if not c:
            if proxy_ret is not None:
                returns[sym] = proxy_ret
                notes.append(f"{sym}: pas en base; utilisation du proxy {proxy_symbol}.")
            else:
                returns[sym] = 0.0
                notes.append(f"{sym}: pas en base; retour 0.0.")
            continue

        r = _cumulative_return_for_crypto(c, start, end)
        if r is None:
            if proxy_ret is not None:
                returns[sym] = proxy_ret
                notes.append(f"{sym}: données manquantes; proxy {proxy_symbol}.")
            else:
                returns[sym] = 0.0
                notes.append(f"{sym}: données manquantes; retour 0.0.")
        else:
            returns[sym] = r

    return returns

def _cumulative_return_for_crypto(crypto: Crypto, start, end) -> float | None:
    """
    Rendement cumulé entre start et end pour une crypto via CryptoInfo, agrégé en médiane horaire.
    Retourne None si pas de data.
    """
    qs = CryptoInfo.objects.filter(
        crypto=crypto,
        timestamp__gte=start,
        timestamp__lte=end,
        current_price__isnull=False
    ).order_by('timestamp')

    if not qs.exists():
        return None

    df = pd.DataFrame([{"timestamp": r.timestamp, "price": float(r.current_price)} for r in qs])
    df = df.set_index("timestamp").sort_index()

    # Agrégation en médiane horaire pour aligner les séries
    df_hourly = df.resample("1H").median().dropna()
    if df_hourly.empty:
        return None

    # Rendement cumulé simple
    start_price = df_hourly["price"].iloc[0]
    end_price = df_hourly["price"].iloc[-1]
    if start_price <= 0 or end_price <= 0:
        return None

    return float(end_price / start_price - 1.0)

# ---------- Application du scénario ----------

def compute_portfolio_base_value_and_weights(portfolio: Portfolio) -> Tuple[float, Dict[str, float], List[str]]:
    """
    Base_value du portefeuille et poids par symbole, via derniers prix CryptoInfo.
    Si PortfolioPerformance existe, on peut l’utiliser comme base_value (facultatif),
    mais on recalcule quand même les poids avec les derniers prix.
    """
    notes: List[str] = []

    # Dernière performance connue (optionnel comme base "officielle")
    latest_perf = portfolio.performances.order_by('-timestamp').first()
    perf_value = float(latest_perf.value) if latest_perf else None

    holdings = list(portfolio.holdings.select_related('crypto').all())
    cryptos = [h.crypto for h in holdings]

    price_map = get_latest_prices_for_cryptos(cryptos, lookback_days=7)

    per_asset_value: Dict[str, float] = {}
    for h in holdings:
        sym = h.crypto.symbol
        px = price_map.get(sym)
        if px is None:
            notes.append(f"Aucun prix récent pour {sym} (7j). Ignoré dans la valeur.")
            continue
        val = float(h.quantity) * float(px)
        if val > 0:
            per_asset_value[sym] = per_asset_value.get(sym, 0.0) + val

    base_value = sum(per_asset_value.values())
    if base_value == 0:
        raise ValueError("Impossible de valoriser le portefeuille (pas de prix récents).")

    weights = {sym: v / base_value for sym, v in per_asset_value.items()}

    # Si tu préfères utiliser perf_value comme base affichée :
    if perf_value and perf_value > 0:
        # On garde les poids du jour (plus réalistes), mais la base affichée de la perf.
        return perf_value, weights, notes

    return base_value, weights, notes

def apply_stress_to_portfolio(portfolio: Portfolio, scenario: Dict) -> Dict:
    """
    Applique un scénario (uniform/factor/historical) au portefeuille.
    Retourne base_value, stressed_value, pnl, pnl_pct et contributions.
    """
    base_value, weights, notes = compute_portfolio_base_value_and_weights(portfolio)
    symbols = list(weights.keys())

    sc_type = scenario.get("type")
    params = scenario.get("params", {}) or {}

    # Déterminer les rendements par actif
    if sc_type == "uniform":
        r_uniform = float(params.get("return", 0.0))
        r_by_asset = {s: r_uniform for s in symbols}

    elif sc_type == "factor":
        default_r = float(params.get("default", 0.0))
        # mapping direct par symbole
        r_by_asset = {s: float(params.get(s, default_r)) for s in symbols}

        # Support optionnel des groupes: {"groups": {...}, "map": {...}, "default_group": "..."}
        if "groups" in params and "map" in params:
            gmap = params["map"]            # symbol -> group
            groups = params["groups"]       # group -> return
            default_group = params.get("default_group")
            for s in symbols:
                if s in gmap:
                    grp = gmap[s]
                    if grp in groups:
                        r_by_asset[s] = float(groups[grp])
                    elif default_group and default_group in groups:
                        r_by_asset[s] = float(groups[default_group])

    elif sc_type == "historical":
        start = pd.to_datetime(params.get("start"))
        end = pd.to_datetime(params.get("end"))
        proxy = params.get("proxy")
        if pd.isna(start) or pd.isna(end):
            raise ValueError("Paramètres historiques invalides: start/end requis.")
        r_by_asset = historical_returns_for_symbols(symbols, start, end, proxy_symbol=proxy, notes=notes)

    else:
        raise ValueError("Type de scénario non supporté.")

    # Agrégation
    pnl_pct = sum(weights[s] * r_by_asset.get(s, 0.0) for s in symbols)
    stressed_value = base_value * (1.0 + pnl_pct)

    by_asset = [
        {
            "symbol": s,
            "weight": weights[s],
            "return": r_by_asset.get(s, 0.0),
            "contribution": weights[s] * r_by_asset.get(s, 0.0),
        }
        for s in symbols
    ]

    return {
        "portfolio_id": portfolio.id,
        "scenario": {
            "name": scenario.get("name", "ad-hoc"),
            "type": sc_type,
        },
        "base_value": float(base_value),
        "stressed_value": float(stressed_value),
        "pnl": float(stressed_value - base_value),
        "pnl_pct": float(pnl_pct),
        "by_asset": by_asset,
        "notes": notes,
    }
