# core/business/risk/pricing.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple
from datetime import timedelta
import numpy as np
import pandas as pd

from .simulate import (
    to_hourly_median, log_returns_pct, simulate_with_ngarch
)

HOURS_PER_YEAR = 24 * 365

@dataclass
class OptionPricingParams:
    symbol: str            # "BTC"
    option_type: str       # "call" | "put"
    strike: float          # K
    risk_free: float       # r annualisé (ex: 0.02 pour 2%)
    horizon_hours: int     # nombre d'heures jusqu’à maturité
    n_sims: int            # nb chemins Monte Carlo (capé côté vue)

def _gbm_fallback_paths(last_price: float, mu: float, sigma_pct: float,
                        horizon_hours: int, n_sims: int) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback simple GBM en cas d’échec NGARCH."""
    T = horizon_hours
    # sigma_pct est en % → convertir en volatilité (log-ret en unités naturelles)
    sigma = sigma_pct / 100.0
    dt = 1.0 / HOURS_PER_YEAR
    # bruit normal(0,1)
    z = np.random.normal(size=(n_sims, T))
    prices = np.empty((n_sims, T), dtype=float)
    prices[:, 0] = last_price * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z[:, 0])
    for t in range(1, T):
        prices[:, t] = prices[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z[:, t])
    vol = np.full(T, sigma_pct, dtype=float)  # juste pour cohérence de shape
    return prices, vol

def price_option_mc(
    df_hourly: pd.DataFrame,
    params: OptionPricingParams
) -> Dict[str, Any]:
    """
    df_hourly: DataFrame indexé temps 1H avec colonne 'price'
    Retourne: prix, intervalle de confiance, diagnostiques, etc.
    """
    if len(df_hourly) < 50:
        raise ValueError("Not enough hourly data to fit the model.")

    # log-returns (%) = ln(Pt/Pt-1)*100
    lr_pct = log_returns_pct(df_hourly["price"])
    last_price = float(df_hourly["price"].iloc[-1])

    # Essai NGARCH → si échec, fallback GBM
    used_model = "ngarch"
    try:
        paths, _vol = simulate_with_ngarch(
            logret_pct=lr_pct.values,
            last_price=last_price,
            horizon_hours=params.horizon_hours,
            n_sims=params.n_sims
        )
    except Exception:
        used_model = "gbm_fallback"
        mu = float(np.mean(lr_pct)) / 100.0     # ~ drift horaire
        sigma_pct = float(np.std(lr_pct))       # en %
        paths, _vol = _gbm_fallback_paths(last_price, mu, sigma_pct,
                                          params.horizon_hours, params.n_sims)

    # payoff au dernier pas (maturité)
    ST = paths[:, -1]
    K = params.strike
    if params.option_type.lower() == "call":
        payoff = np.maximum(ST - K, 0.0)
    else:
        payoff = np.maximum(K - ST, 0.0)

    # actualisation
    T_years = params.horizon_hours / HOURS_PER_YEAR
    disc = np.exp(-params.risk_free * T_years)
    price = float(disc * payoff.mean())

    # erreur std MC + IC 95%
    std_mc = float(payoff.std(ddof=1)) * disc
    se = std_mc / np.sqrt(params.n_sims)
    ci95 = (price - 1.96 * se, price + 1.96 * se)

    return {
        "model_used": used_model,
        "last_price": last_price,
        "price": price,
        "ci95": ci95,
        "stderr": se,
        "n_sims": params.n_sims,
        "horizon_hours": params.horizon_hours,
        "risk_free": params.risk_free,
    }


