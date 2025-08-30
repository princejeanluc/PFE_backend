# core/business/risk/simulate.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from scipy.stats import t as student , genpareto

from core.libs.garch import MixtureParams, NGARCHMixMLE, TailParams, build_continuous_student_gpd_mixture

@dataclass
class RiskSimParams:
    symbol: str
    horizon_hours: int  # ex: 72
    n_sims: int         # ex: 200

def to_hourly_median(df: pd.DataFrame, ts_col="timestamp") -> pd.DataFrame:
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.set_index(ts_col).resample("1h").median()
    return df.dropna()

def log_returns_pct(prices: pd.Series) -> pd.Series:
    # log-return (%) = ln(Pt/Pt-1) * 100
    return np.log(prices / prices.shift(1)).mul(100).dropna()

def fit_gpd_tails_auto(residuals, seuils_test_droite=None, seuils_test_gauche=None):
    """
    Ajuste deux lois de Pareto généralisée (GPD) avec détection automatique du seuil
    pour les queues droite et gauche, indépendamment.

    Paramètres
    ----------
    residuals : array-like
        Résidus standardisés ou innovations.

    seuils_test_droite : array-like or None
        Liste des seuils à tester pour la queue droite. Si None, générée automatiquement.

    seuils_test_gauche : array-like or None
        Liste des seuils à tester pour la queue gauche. Si None, générée automatiquement.

    plot : bool
        Si True, affiche les diagnostics graphiques.

    Retour
    ------
    dict : paramètres shape, scale et seuil optimal pour chaque queue.
    """
    residuals = np.asarray(residuals)
    
    # === Queue droite ===
    right_data = residuals[residuals > 0]
    if seuils_test_droite is None:
        seuils_test_droite = np.linspace(np.percentile(right_data, 85), np.percentile(right_data, 99), 20)

    mrl_means_pos = []
    xi_list_pos = []
    scale_list_pos = []
    
    for u in seuils_test_droite:
        excess = right_data[right_data > u] - u
        if len(excess) < 30:  # Éviter des seuils trop extrêmes
            mrl_means_pos.append(np.nan)
            xi_list_pos.append(np.nan)
            scale_list_pos.append(np.nan)
            continue
        mrl_means_pos.append(np.mean(excess))
        xi, loc, scale = genpareto.fit(excess)
        xi_list_pos.append(xi)
        scale_list_pos.append(scale)
    
    # Choix du seuil optimal : début de la stabilité de xi
    xi_array = np.array(xi_list_pos)
    stable_index_pos = np.nanargmin(np.abs(np.gradient(xi_array)))
    u_pos_opt = seuils_test_droite[stable_index_pos]
    
    # Fit final pour la queue droite
    excess_pos_final = right_data[right_data > u_pos_opt] - u_pos_opt
    params_pos = genpareto.fit(excess_pos_final)

    # === Queue gauche ===
    left_data = -residuals[residuals < 0]
    if seuils_test_gauche is None:
        seuils_test_gauche = np.linspace(np.percentile(left_data, 85), np.percentile(left_data, 99), 20)

    mrl_means_neg = []
    xi_list_neg = []
    scale_list_neg = []
    
    for u in seuils_test_gauche:
        excess = left_data[left_data > u] - u
        if len(excess) < 30:
            mrl_means_neg.append(np.nan)
            xi_list_neg.append(np.nan)
            scale_list_neg.append(np.nan)
            continue
        mrl_means_neg.append(np.mean(excess))
        xi, loc, scale = genpareto.fit(excess)
        xi_list_neg.append(xi)
        scale_list_neg.append(scale)
    
    # Choix du seuil optimal : début de la stabilité de xi
    xi_array_neg = np.array(xi_list_neg)
    stable_index_neg = np.nanargmin(np.abs(np.gradient(xi_array_neg)))
    u_neg_opt = seuils_test_gauche[stable_index_neg]
    
    # Fit final pour la queue gauche
    excess_neg_final = left_data[left_data > u_neg_opt] - u_neg_opt
    params_neg = genpareto.fit(excess_neg_final)


    return {
        "right_tail": {
            "threshold": u_pos_opt,
            "xi": params_pos[0],
            "scale": params_pos[2]
        },
        "left_tail": {
            "threshold": -u_neg_opt,
            "xi": params_neg[0],
            "scale": params_neg[2]
        }
    }


# ---- NGARCH + loi mixte : branch point ----

def simulate_with_ngarch(
    logret_pct: np.ndarray,
    last_price: float,
    horizon_hours: int,
    n_sims: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Doit retourner:
      paths: shape (n_sims, T) (prix simulés)
      vol:   shape (T,)       (volatilité prédite par pas)
    Implémentation provisoire (fallback) : modèle simple
    — à remplacer par ton N-GARCH + loi mixte.
    """
    T = horizon_hours
    if len(logret_pct) < 10:
        raise ValueError("Not enough data for fitting")

    gpd_params = fit_gpd_tails_auto(logret_pct)
    df_t, loc_t, scale_t = student.fit(logret_pct)
    mixture_params = MixtureParams(
        df_student = float(df_t),
        left  = TailParams(xi=float(gpd_params['left_tail']['xi']),
                           beta=float(gpd_params['left_tail']['scale']),
                           u=float(gpd_params['left_tail']['threshold'])),
        right = TailParams(xi=float(gpd_params['right_tail']['xi']),
                           beta=float(gpd_params['right_tail']['scale']),
                           u=float(gpd_params['right_tail']['threshold'])),
    )
    f_Z, sample_Z, mix_info = build_continuous_student_gpd_mixture(mixture_params)
    ngarch = NGARCHMixMLE(f_Z, sample_Z, standardize=True).fit(logret_pct)
    vol = ngarch.forecast(horizon=T, mode="mc",n_paths=n_sims)[0]
    eps = np.zeros((n_sims, T))
    for n_sim in range(n_sims):
        eps[n_sim] = ngarch.sample_Z_std(T)*np.sqrt(vol)

    # convertir en log-return naturels avant exponentielle: (pct/100)
    prices = np.empty((n_sims, T), dtype=float)
    prices[:, 0] = last_price * np.exp(eps[:, 0] / 100.0)
    for t in range(1, T):
        prices[:, t] = prices[:, t-1] * np.exp(eps[:, t] / 100.0)

    return prices, vol

def compute_metrics_from_paths(paths: np.ndarray) -> Dict[str, float]:
    """
    Calcule metrics sur les rendements terminaux (du dernier point vs premier point futur).
    Tu peux passer aux P&L ou quantiles intermédiaires si tu préfères.
    """
    # Rendement terminal (%) par simulation
    r_terminal = (paths[:, -1] / paths[:, 0] - 1.0)
    mu = float(np.mean(r_terminal))
    sd = float(np.std(r_terminal)) + 1e-12

    # VaR / ES à 95%
    q = 0.05
    var_95 = float(np.quantile(r_terminal, q))      # VaR négative = perte
    es_95 = float(r_terminal[r_terminal <= var_95].mean()) if np.any(r_terminal <= var_95) else var_95

    # Sharpe non annualisé (à toi d’annualiser selon l’horizon si besoin)
    sharpe = float(mu / sd)

    return {"var_95": var_95, "es_95": es_95, "sharpe": sharpe}

def build_history_for_response(df_hourly: pd.DataFrame, keep_last_hours=24*3) -> Dict[str, Any]:
    if len(df_hourly) == 0:
        return {"timestamps": [], "prices": []}
    tail = df_hourly.tail(keep_last_hours)
    return {
        "timestamps": [ts.isoformat() for ts in tail.index.to_pydatetime()],
        "prices": [float(p) for p in tail["price"].values],
    }
