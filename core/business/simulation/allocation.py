# core.business.simulation.allocation

from core.models import Crypto, CryptoInfo, Holding, Portfolio, PortfolioPerformance
from django.utils.timezone import now , make_aware
from datetime import timedelta , datetime
import pandas as pd
import numpy as np
import cvxpy as cp

PERIOD_DAY = 30 # nb jours


def compute_optimal_weights(price_df: pd.DataFrame, objective: str = "sharpe") -> dict:
    returns = np.log(price_df / price_df.shift(1)).dropna()
    mu = returns.mean().values
    Sigma = returns.cov().values
    n = len(mu)

    w = cp.Variable(n)
    constraints = [cp.sum(w) == 1, w >= 0]

    if objective == "sharpe":
        risk_free = 0.0
        excess = mu - risk_free
        problem = cp.Problem(cp.Maximize(excess @ w / cp.sqrt(cp.quad_form(w, Sigma))), constraints)
    elif objective == "min_volatility":
        problem = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)), constraints)
    elif objective == "max_return":
        problem = cp.Problem(cp.Maximize(mu @ w), constraints)
    else:
        raise ValueError(f"Objectif non reconnu : {objective}")

    problem.solve()
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

def create_performance_series(portfolio):
    """
    Supprime les performances existantes et crée une nouvelle série horaire
    basée sur les valeurs simulées du portefeuille.
    """
    # 0. Nettoyage
    PortfolioPerformance.objects.filter(portfolio=portfolio).delete()

    # 1. Paramètres de la simulation
    start_dt = make_aware(datetime.combine(portfolio.holding_start, datetime.min.time()))
    end_dt = make_aware(datetime.combine(portfolio.holding_end, datetime.max.time()))
    now_dt = now()
    final_dt = min(end_dt, now_dt)

    # 2. Récupérer tous les prix agrégés par heure sur la période pour les cryptos du portefeuille
    symbols = [h.crypto.symbol for h in portfolio.holdings.all()]
    price_data = {}
    for h in portfolio.holdings.all():
        infos = CryptoInfo.objects.filter(
            crypto=h.crypto,
            timestamp__gte=start_dt - timedelta(hours=1),  # marge de sécurité
            timestamp__lte=final_dt
        ).order_by("timestamp")

        if not infos.exists():
            continue

        df = pd.DataFrame.from_records([
            {"timestamp": i.timestamp, "price": i.current_price} for i in infos if i.current_price
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df = df.resample("1h").median().dropna()
        price_data[h.crypto.symbol] = df["price"]

    if not price_data:
        print("Pas de données de prix disponibles.")
        return

    df_prices = pd.DataFrame(price_data).dropna()

    # 3. Calculer la valeur du portefeuille à chaque heure
    holdings = {h.crypto.symbol: h for h in portfolio.holdings.all()}
    df_value = pd.DataFrame()
    df_value['value'] = df_prices.apply(
        lambda row: sum(
            holdings[sym].quantity * row[sym] for sym in df_prices.columns
        ), axis=1
    )

    # 4. Calculer les rendements log
    df_value['log_return'] = np.log(df_value['value'] / df_value['value'].shift(1))
    df_value.dropna(inplace=True)

    # 5. Création des performances
    for timestamp, row in df_value.iterrows():
        history = df_value[df_value.index <= timestamp]
        values = history['value']
        returns = history['log_return']
        performance = PortfolioPerformance(
            portfolio=portfolio,
            timestamp=timestamp,
            value=safe(row['value']),
            cumulative_return=safe((row['value'] / df_value['value'].iloc[0]) - 1),
            volatility=safe(returns.std() if len(returns) > 1 else None),
            sharpe_ratio=safe(returns.mean() / returns.std()) if len(returns) > 1 and returns.std() != 0 else None,
            drawdown=safe((row['value'] / values.cummax().loc[timestamp]) - 1),
            sortino_ratio=safe(returns.mean() / returns[returns < 0].std()) if len(returns[returns < 0]) > 0 else None,
            expected_shortfall=safe(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else None,
            value_at_risk=safe(returns.quantile(0.05)),
            information_ratio=safe(returns.mean() / returns.std()) if len(returns) > 1 and returns.std() != 0 else None
        )
        performance.save()

    print(f"{len(df_value)} performances créées pour le portefeuille {portfolio.name}")
