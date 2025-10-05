# core.business.simulation.allocation

import math
from core.models import Crypto, CryptoInfo, Holding, Portfolio, PortfolioPerformance
from django.utils.timezone import now , make_aware
from datetime import timedelta , datetime
from django.db import transaction
from math import sqrt
import logging
import pandas as pd
import numpy as np
import cvxpy as cp

PERIOD_DAY = 30 # nb jours


def nearest_psd(Sigma, eps=1e-8):
    # symétriser + petit ridge pour éviter la quasi-singularité
    S = 0.5 * (Sigma + Sigma.T)
    # ridge
    return S + eps * np.eye(S.shape[0])

def compute_optimal_weights(price_df: pd.DataFrame, objective: str = "sharpe",
                            risk_aversion: float = 10.0, target_return: float | None = None) -> dict:
    returns = np.log(price_df / price_df.shift(1)).dropna()
    mu = returns.mean().values
    Sigma = returns.cov().values
    n = len(mu)

    # garde-fous
    if n == 0:
        raise ValueError("Pas assez d'actifs.")
    if np.allclose(Sigma, 0):
        # fallback trivial : tout sur le meilleur mu ou répartition égale
        w = np.ones(n) / n
        return {symbol: float(round(wi, 4)) for symbol, wi in zip(price_df.columns, w)}

    Sigma = nearest_psd(Sigma, eps=1e-6)

    w = cp.Variable(n)
    constraints = [cp.sum(w) == 1, w >= 0]

    if objective == "sharpe":
        # ✅ DCP-friendly : trade-off moyen-variance
        obj = cp.Maximize(mu @ w - risk_aversion * cp.quad_form(w, Sigma))
        prob = cp.Problem(obj, constraints)

    elif objective == "min_volatility":
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)), constraints)

    elif objective == "max_return":
        prob = cp.Problem(cp.Maximize(mu @ w), constraints)

    elif objective == "target_return":
        # ✅ DCP : variance minimale sous contrainte de rendement
        R = float(target_return) if target_return is not None else float(np.mean(mu))
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)), constraints + [mu @ w >= R])

    else:
        raise ValueError(f"Objectif non reconnu : {objective}")

    prob.solve(solver=cp.OSQP, verbose=False)  # ou SCS
    if w.value is None:
        raise ValueError("Échec de l'optimisation : poids non définis")
    return {symbol: round(float(weight), 4) for symbol, weight in zip(price_df.columns, w.value)}


def run_markowitz_allocation(portfolio: Portfolio):
    end_date = now()
    start_date = end_date - timedelta(days=PERIOD_DAY)

    crypto_ids = portfolio.holdings.values_list("crypto__id", flat=True)
    cryptos = Crypto.objects.filter(id__in=crypto_ids)

    infos = CryptoInfo.objects.filter(
        crypto__in=cryptos,
        timestamp__range=(start_date, end_date),
        current_price__isnull=False
    ).order_by("timestamp")

    # Construction du DataFrame brut
    records = [
        {"timestamp": info.timestamp, "symbol": info.crypto.symbol, "price": info.current_price}
        for info in infos
        if info.timestamp and info.current_price is not None
    ]

    if not records:
        raise ValueError("Aucune donnée de prix disponible pour les cryptos du portefeuille.")

    df = pd.DataFrame(records)

    if "timestamp" not in df.columns or df.empty:
        raise ValueError("Les données récupérées sont invalides ou incomplètes (colonne timestamp manquante).")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    # Pivot: symboles en colonnes, index = timestamps, valeurs = prix
    pivot = df.pivot_table(index="timestamp", columns="symbol", values="price", aggfunc="median")

    df_hourly = pivot.resample("1h").median().dropna(how='any')

    if df_hourly.empty:
        raise ValueError("Pas de données disponibles après agrégation horaire pour Markowitz.")

    # Objectif depuis le portfolio
    objective = portfolio.objective or "sharpe"
    weights = compute_optimal_weights(df_hourly, objective=objective)

    for symbol, weight in weights.items():
        crypto = Crypto.objects.get(symbol=symbol)
        Holding.objects.update_or_create(
            portfolio=portfolio,
            crypto=crypto,
            defaults={
                "allocation_percentage": round(weight * 100, 2),
                "quantity": 0  # sera calculé plus tard via initialize_quantities
            }
        )


def initialize_quantities(portfolio:Portfolio):
    """
    Calcule les quantités initiales de chaque crypto en fonction du budget et des prix à holding_start.
    """

    holdings = portfolio.holdings.all()
    start_dt = make_aware(datetime.combine(portfolio.holding_start, datetime.min.time()))

    for h in holdings:
        if h.quantity > 0:
            continue  # déjà défini

        # On récupère le prix de la crypto au début de la période (heure la plus proche avant ou égale)
        infos = CryptoInfo.objects.filter(
            crypto=h.crypto,
            timestamp__lte=start_dt,
            current_price__isnull=False
        ).order_by('-timestamp')

        if not infos.exists():
            print(f"[Warning] Pas de données pour {h.crypto.symbol} à {start_dt}")
            continue

        price = infos.first().current_price
        allocation_usd = (h.allocation_percentage / 100) * portfolio.initial_budget
        quantity = allocation_usd / price

        h.quantity = quantity
        h.purchase_price = price
        h.save()


def run_markowitz_allocation(portfolio: Portfolio):
    end_date = now()
    start_date = end_date - timedelta(days=PERIOD_DAY)
    crypto_ids = portfolio.holdings.values_list("crypto__id", flat=True)
    print("cryptoId", crypto_ids)
    cryptos = Crypto.objects.filter(id__in=crypto_ids)
    print("cryptos", cryptos)
    infos = CryptoInfo.objects.filter(
        crypto__in=cryptos,
        timestamp__range=(start_date, end_date),
        current_price__isnull=False
    ).order_by("timestamp")
    print("infos",infos)
    # Construction du DataFrame brut
    records = [
        {"timestamp": info.timestamp, "symbol": info.crypto.symbol, "price": info.current_price}
        for info in infos
        if info.timestamp and info.current_price is not None
    ]

    if not records:
        raise ValueError("Aucune donnée de prix disponible pour les cryptos du portefeuille.")

    df = pd.DataFrame.from_records(records)

    if "timestamp" not in df.columns or df.empty:
        print("⚠️ DataFrame vide ou colonne 'timestamp' manquante.")
        print("Contenu brut :", records[:3])
        raise ValueError("Les données récupérées sont invalides.")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    # Pivot: symboles en colonnes, index = timestamps, valeurs = prix
    pivot = df.pivot_table(index="timestamp", columns="symbol", values="price", aggfunc="median")

    df_hourly = pivot.resample("1h").median().dropna(how='any')

    if df_hourly.empty:
        raise ValueError("Pas de données disponibles après agrégation horaire pour Markowitz.")

    # Objectif depuis le portfolio
    objective = portfolio.objective or "sharpe"
    weights = compute_optimal_weights(df_hourly, objective=objective)

    for symbol, weight in weights.items():
        crypto = Crypto.objects.get(symbol=symbol)
        Holding.objects.update_or_create(
            portfolio=portfolio,
            crypto=crypto,
            defaults={
                "allocation_percentage": round(weight * 100, 2),
                "quantity": 0  # sera calculé plus tard via initialize_quantities
            }
        )

def safe(x):
    return None if pd.isna(x) else float(x)

EPS = 1e-12

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def to_nullable_number(x):
    """
    Convertit x en float ou None si x est None/NaN/inf ou convert impossible.
    """
    if x is None:
        return None
    try:
        xf = float(x)
    except Exception:
        return None
    if math.isnan(xf) or math.isinf(xf):
        return None
    return xf
def create_performance_series(portfolio):
    """
    Version optimisée :
    - lecture limitée des prix par crypto (fenêtre start..final_dt)
    - pivot unique en DataFrame horaires
    - calcul vectoriel des métriques (expanding / cumsum)
    - bulk_create pour écrire en base
    """
    logging.info("La création des performances a commencé")
    # 0. Nettoyage
    PortfolioPerformance.objects.filter(portfolio=portfolio).delete()

    # 1. Paramètres
    start_dt = make_aware(datetime.combine(portfolio.holding_start, datetime.min.time()))
    end_dt = make_aware(datetime.combine(portfolio.holding_end, datetime.max.time()))
    now_dt = now()
    final_dt = min(end_dt, now_dt)

    # Récupérer symboles et holdings en une fois
    holdings_qs = list(portfolio.holdings.select_related("crypto").all())
    if not holdings_qs:
        print("Aucun holding pour ce portefeuille.")
        return

    symbols = [h.crypto.symbol for h in holdings_qs]

    # 2. Charger les séries horaires pour toutes les cryptos (une requête par crypto)
    #    -> si tu as MarketSnapshot (horaire), prefère l'utiliser ici.
    price_series = {}
    for h in holdings_qs:
        # Filtre SQL limité à la fenêtre utile
        qs = CryptoInfo.objects.filter(
            crypto=h.crypto,
            timestamp__gte=start_dt - timedelta(hours=1),
            timestamp__lte=final_dt,
            current_price__isnull=False
        ).order_by("timestamp").values_list("timestamp", "current_price")

        rows = list(qs)
        if not rows:
            continue

        df = pd.DataFrame(rows, columns=["timestamp", "price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()

        # Resample horaire médiane (si tes données sont déjà horaires, c'est quasi no-op)
        df = df.resample("1h").median().dropna()
        price_series[h.crypto.symbol] = df["price"]

    if not price_series:
        print("Pas de données de prix disponibles.")
        return

    # Construire dataframe large (index=timestamp, colonnes=symbol)
    df_prices = pd.DataFrame(price_series).dropna().sort_index()
    if df_prices.empty:
        print("Pas assez de données horaires synchrones.")
        return

    # 3. Calculer la valeur du portefeuille vectorisée
    # holdings_map : symbol -> quantity
    holdings_map = {h.crypto.symbol: float(h.quantity) for h in holdings_qs}
    # S'assurer que toutes colonnes ont un holding
    used_symbols = [c for c in df_prices.columns if c in holdings_map]
    df_prices = df_prices[used_symbols]

    # multiply each column by quantity
    for sym in used_symbols:
        df_prices[sym] = df_prices[sym] * holdings_map[sym]

    df_value = pd.DataFrame(index=df_prices.index)
    df_value["value"] = df_prices.sum(axis=1)

    # 4. Rendements log (horaire)
    df_value["log_return"] = np.log(df_value["value"] / df_value["value"].shift(1))
    df_value = df_value.dropna()
    if df_value.empty:
        print("Pas de rendements calculables.")
        return

    # 5. Vectorisation des métriques cumulatives (expanding)
    returns = df_value["log_return"]

    # cumul mean and std (expanding)
    cum_mean = returns.expanding().mean()
    cum_std = returns.expanding().std(ddof=0)  # ddof=0 pour population-like

    # sharpe (mean/std), safe handling
    sharpe = (cum_mean / cum_std).replace([np.inf, -np.inf], np.nan)

    # cumulative return relative au premier point
    first_val = df_value["value"].iloc[0]
    cumulative_return = df_value["value"] / (first_val + EPS) - 1.0
    cumulative_return = cumulative_return.fillna(method=0)  # rien
    # forcer le premier point si il est NaN
    if not pd.isna(cumulative_return.iloc[0]):
        pass
    else:
        cumulative_return.iloc[0] = 0.0
    # drawdown = current / running_max - 1
    running_max = df_value["value"].cummax()
    drawdown = df_value["value"] / (running_max + EPS) - 1.0
    

    # Value at Risk (expanding quantile 5%)
    try:
        var05 = returns.expanding().quantile(0.05)
    except Exception:
        # fallback: compute using rolling-ish approach (slower) or set NaN
        var05 = pd.Series(index=returns.index, data=np.nan)

    # Expected Shortfall (ES): cumulative mean of negative returns
    neg_vals = returns.where(returns < 0, 0.0)
    neg_sum_cum = neg_vals.cumsum()
    neg_count_cum = (returns < 0).cumsum()
    expected_shortfall = (neg_sum_cum / (neg_count_cum.replace(0, np.nan))).fillna(np.nan)

    # Downside std: compute cumulative sum of squares for negatives and derive std
    neg_sq = (returns.where(returns < 0, 0.0) ** 2)
    neg_sq_cum = neg_sq.cumsum()
    # variance = E[X^2] - mean^2 on negatives
    neg_mean = expected_shortfall  # already mean of negatives
    downside_var = (neg_sq_cum / neg_count_cum.replace(0, np.nan)) - (neg_mean ** 2)
    downside_var = downside_var.clip(lower=0.0)
    downside_std = np.sqrt(downside_var).replace([np.inf, -np.inf], np.nan)

    # Information ratio: here on cumulative basis same as sharpe (can be adapted)
    information_ratio = sharpe

    # sortino = mean / downside_std (safe)
    sortino = (cum_mean / downside_std).replace([np.inf, -np.inf], np.nan)

    # 6. Préparer les objets PortfolioPerformance (bulk create)
    objs = []
    # iterate index once (once) to build instances
    # récupérer la classe locale
    PP = PortfolioPerformance
    for ts in df_value.index:
        val = safe_float(df_value.at[ts, "value"])
        if val is None:
            continue
        # fetch vector metrics at ts
        cr = safe_float(cumulative_return.at[ts]) if ts in cumulative_return.index else None
        vol = safe_float(cum_std.at[ts]) if ts in cum_std.index else None
        sr = safe_float(sharpe.at[ts]) if ts in sharpe.index else None
        dd = safe_float(drawdown.at[ts]) if ts in drawdown.index else None
        sot = safe_float(sortino.at[ts]) if ts in sortino.index else None
        es = safe_float(expected_shortfall.at[ts]) if ts in expected_shortfall.index else None
        varv = safe_float(var05.at[ts]) if ts in var05.index else None
        ir = safe_float(information_ratio.at[ts]) if ts in information_ratio.index else None

        p = PP(
            portfolio=portfolio,
            timestamp=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
            value=val,
            cumulative_return=cr,
            volatility=vol,
            sharpe_ratio=sr,
            drawdown=dd,
            sortino_ratio=sot,
            expected_shortfall=es,
            value_at_risk=varv,
            information_ratio=ir
        )
        objs.append(p)

    if not objs:
        print("Aucun enregistrement de performance à créer.")
        return

    # 7. Bulk create with transaction for safety
    BATCH = 1000
    with transaction.atomic():
        PortfolioPerformance.objects.bulk_create(objs, batch_size=BATCH)

    print(f"{len(objs)} performances créées pour le portefeuille {portfolio.name}")