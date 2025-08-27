# core/management/commands/backtest_metrics.py
import os
import math
import json
import numpy as np
import pandas as pd
from datetime import datetime
from django.core.management.base import BaseCommand, CommandError
from django.db.models import Q
from django.utils import timezone

from core.models import Crypto, MarketSnapshot, Prediction  # ajuste si ton app n'est pas "core"

# ----------------------------
#   Métriques de backtest
# ----------------------------

def max_drawdown(cum_returns: pd.Series) -> float:
    """
    Max Drawdown sur la courbe de valeur cumulée (1 + ret). Ex: Series du type (1+r).cumprod().
    Retourne une valeur négative ou 0 (ex: -0.27 = -27%).
    """
    if cum_returns.empty:
        return 0.0
    peak = cum_returns.cummax()
    dd = cum_returns / peak - 1.0
    return dd.min()

def annualize_return(ret_series: pd.Series, periods_per_year: int = 24*365) -> float:
    """Annualisation à partir de rendements périodiques (ex: horaires)."""
    if ret_series.empty:
        return 0.0
    growth = (1.0 + ret_series).prod()
    n = ret_series.shape[0]
    if n == 0:
        return 0.0
    # Taux moyen par période -> annualisation
    avg_per_period = growth ** (1.0 / n) - 1.0
    return (1.0 + avg_per_period) ** periods_per_year - 1.0

def annualized_vol(ret_series: pd.Series, periods_per_year: int = 24*365) -> float:
    """Vol annualisée (écart-type * sqrt(periods_per_year))."""
    if ret_series.empty:
        return 0.0
    return float(ret_series.std(ddof=1) * math.sqrt(periods_per_year))

def sharpe_ratio(ret_series: pd.Series, rf_per_period: float = 0.0, periods_per_year: int = 24*365) -> float:
    """
    Sharpe annualisé. Par défaut rf=0.0 par période (tu peux passer un taux horaire).
    """
    if ret_series.empty:
        return 0.0
    excess = ret_series - rf_per_period
    mu = excess.mean()
    sigma = excess.std(ddof=1)
    if sigma == 0 or np.isnan(sigma):
        return 0.0
    return float(mu / sigma * math.sqrt(periods_per_year))

def information_ratio(strategy_ret: pd.Series, benchmark_ret: pd.Series) -> float:
    """
    IR = mean(active) / std(active), active = strategy - benchmark (même fréquence).
    """
    if strategy_ret.empty or benchmark_ret.empty:
        return 0.0
    # Aligner les index
    strategy_ret, benchmark_ret = strategy_ret.align(benchmark_ret, join="inner")
    active = strategy_ret - benchmark_ret
    den = active.std(ddof=1)
    if den == 0 or np.isnan(den):
        return 0.0
    return float(active.mean() / den)

def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    if y_true.empty:
        return 0.0
    y_true, y_pred = y_true.align(y_pred, join="inner")
    return float(np.abs(y_true - y_pred).mean())

def mse(y_true: pd.Series, y_pred: pd.Series) -> float:
    if y_true.empty:
        return 0.0
    y_true, y_pred = y_true.align(y_pred, join="inner")
    return float(((y_true - y_pred) ** 2).mean())

def mcfd_directional_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    MCFD (ici comme "Mean Correct Forecast Direction"): part des périodes
    où sign(pred) == sign(true). Donne une précision directionnelle entre 0 et 1.
    """
    if y_true.empty:
        return 0.0
    y_true, y_pred = y_true.align(y_pred, join="inner")
    s_true = np.sign(y_true)
    s_pred = np.sign(y_pred)
    mask = (s_true != 0)  # ignore les périodes neutres si besoin
    if mask.sum() == 0:
        return 0.0
    return float((s_true[mask] == s_pred[mask]).mean())

def mftr_mean_trade_return(true_ret: pd.Series, pred_ret: pd.Series, cost_per_trade: float = 0.0) -> float:
    """
    MFTR (ici comme "Mean Financial Trade Return"): moyenne des rendements de la stratégie
    directionnelle simple: signal = sign(pred), strat_ret = signal * true_ret - cost.
    """
    if true_ret.empty:
        return 0.0
    true_ret, pred_ret = true_ret.align(pred_ret, join="inner")
    signal = np.sign(pred_ret).astype(float)
    strat = signal * true_ret - cost_per_trade * (signal != 0).astype(float)
    return float(strat.mean())

# ----------------------------
#   Extraction & backtest
# ----------------------------

def load_backtest_pairs(crypto_symbols=None, model_names=None, start=None, end=None):
    """
    Récupère (Prediction, MarketSnapshot) alignés par (crypto, predicted_date == timestamp).
    Renvoie un DataFrame de base niveau 'trade' (ligne = une période pour un modèle/crypto).
    Colonnes: crypto, model_name, ts, pred_log_return, true_ret, benchmark_ret ...
    """
    pred_qs = Prediction.objects.all()

    if crypto_symbols:
        pred_qs = pred_qs.filter(crypto__symbol__in=crypto_symbols)
    if model_names:
        pred_qs = pred_qs.filter(model_name__in=model_names)
    if start:
        pred_qs = pred_qs.filter(predicted_date__gte=start)
    if end:
        pred_qs = pred_qs.filter(predicted_date__lte=end)

    # On passe par une liste de dicts pour créer un DF
    preds = list(pred_qs.values(
        'crypto__symbol', 'crypto__id', 'model_name', 'predicted_date', 'predicted_log_return'
    ))

    if not preds:
        return pd.DataFrame()

    df_pred = pd.DataFrame.from_records(preds).rename(columns={
        'crypto__symbol': 'symbol',
        'crypto__id': 'crypto_id',
        'predicted_date': 'ts',
        'predicted_log_return': 'pred_log_ret'
    })

    # Charger les snapshots correspondants à (crypto, ts) pour récupérer le true return
    # NB: MarketSnapshot.hourly_return = rendement simple sur la période horaire
    #     On prend la valeur où MarketSnapshot.timestamp == Prediction.predicted_date
    #     (ajuste si ta convention est t -> t+1)
    snap_qs = MarketSnapshot.objects.filter(
        crypto__id__in=df_pred['crypto_id'].unique(),
        timestamp__in=df_pred['ts'].unique()
    ).values('crypto__id', 'timestamp', 'hourly_return')

    df_snap = pd.DataFrame.from_records(snap_qs).rename(columns={
        'crypto__id': 'crypto_id',
        'timestamp': 'ts',
        'hourly_return': 'true_ret'
    })

    # Merge inner (on garde uniquement les paires alignées)
    df = pd.merge(df_pred, df_snap, on=['crypto_id', 'ts'], how='inner')
    # Benchmark "buy & hold" = true_ret (c’est le marché pur sur la période)
    df['benchmark_ret'] = df['true_ret']

    # Ajout d'un identifiant pratique
    df['pair'] = df['symbol'].astype(str) + ' | ' + df['model_name'].astype(str)
    return df.sort_values(['symbol', 'model_name', 'ts']).reset_index(drop=True)

def compute_strategy_returns(df_pair: pd.DataFrame, use_log=False, cost_per_trade=0.0) -> pd.DataFrame:
    df = df_pair.sort_values(['symbol', 'model_name', 'ts']).copy()

    # Position binaire : -1 / 0 / +1
    df['position'] = np.sign(df['pred_log_ret']).astype(float)

    # Turnover = quantité tradée (variation de position)
    df['turnover'] = df.groupby(['symbol', 'model_name'])['position'] \
                       .diff().abs().fillna(df['position'].abs())

    # Rendement de stratégie net coûts (selon ton choix d’alignement)
    if use_log:
        # Variante log si tu utilises des log-returns "vrais" pour le vrai r_t
        df['strat_ret'] = df['position'] * df['true_ret'] - cost_per_trade * df['turnover']
    else:
        df['strat_ret'] = df['position'] * df['true_ret'] - cost_per_trade * df['turnover']

    # Cumuls par (symbol, model)
    grp = [df['symbol'], df['model_name']]
    df['cum_value'] = (1.0 + df['strat_ret']).groupby(grp).cumprod()
    df['benchmark_cum'] = (1.0 + df['benchmark_ret']).groupby(grp).cumprod()

    return df


def aggregate_metrics(trades_df: pd.DataFrame, periods_per_year=24*365, rf_per_period=0.0) -> pd.DataFrame:
    """
    Agrège par (symbol, model_name) pour produire un tableau de métriques.
    """
    rows = []
    for (symbol, model), g in trades_df.groupby(['symbol', 'model_name']):
        strat_ret = g['strat_ret']
        bench_ret = g['benchmark_ret']
        true_ret = g['true_ret']
        pred_log = g['pred_log_ret']

        metrics = {
            'symbol': symbol,
            'model_name': model,
            # Performance
            'MFTR': mftr_mean_trade_return(true_ret=true_ret, pred_ret=pred_log, cost_per_trade=0.0),
            'ARR': annualize_return(strat_ret, periods_per_year),
            'AVol': annualized_vol(strat_ret, periods_per_year),
            'MDD': max_drawdown((1.0 + strat_ret).cumprod()),
            'Sharpe': sharpe_ratio(strat_ret, rf_per_period, periods_per_year),
            'IR': information_ratio(strat_ret, bench_ret),
            # Direction / erreurs
            'MCFD': mcfd_directional_accuracy(true_ret, pred_log),  # proportion de directions correctes
            'MAE': mae(true_ret, pred_log),
            'MSE': mse(true_ret, pred_log),
            # Comptes
            'n_trades': int(g.shape[0]),
            'start': g['ts'].min(),
            'end': g['ts'].max(),
        }
        rows.append(metrics)
    return pd.DataFrame(rows).sort_values(['symbol', 'model_name']).reset_index(drop=True)

# ----------------------------
#   Commande Django
# ----------------------------

class Command(BaseCommand):
    help = "Backtest des prédictions par crypto & modèle et export des métriques en DataFrames (CSV/Parquet)."

    def add_arguments(self, parser):
        parser.add_argument("--symbols", type=str, default="",
                            help="Liste de symboles séparés par des virgules (ex: BTC,ETH). Vide = tous.")
        parser.add_argument("--models", type=str, default="",
                            help="Liste de model_name séparés par des virgules. Vide = tous.")
        parser.add_argument("--start", type=str, default="", help="Début (ISO, ex: 2025-05-01T00:00:00)")
        parser.add_argument("--end", type=str, default="", help="Fin (ISO)")
        parser.add_argument("--use-log", action="store_true", help="Comparer en log-returns pour la stratégie.")
        parser.add_argument("--cost", type=float, default=0.0, help="Coût par trade (période).")
        parser.add_argument("--outdir", type=str, default="backtests_out", help="Dossier de sortie.")
        parser.add_argument("--format", type=str, default="csv", choices=["csv", "parquet"], help="Format export.")
        parser.add_argument("--periods-per-year", type=int, default=24*365, help="Périodes/an (8760 pour horaire).")
        parser.add_argument("--rf-per-period", type=float, default=0.0, help="Taux sans risque par période.")

    def handle(self, *args, **opts):
        symbols = [s.strip() for s in opts["symbols"].split(",") if s.strip()] or None
        models = [m.strip() for m in opts["models"].split(",") if m.strip()] or None
        start = self._parse_dt(opts["start"])
        end = self._parse_dt(opts["end"])
        use_log = bool(opts["use_log"])
        cost = float(opts["cost"])
        outdir = opts["outdir"]
        fmt = opts["format"]
        ppy = int(opts["periods_per_year"])
        rf = float(opts["rf_per_period"])

        df_base = load_backtest_pairs(symbols, models, start, end)
        if df_base.empty:
            self.stdout.write(self.style.WARNING("Aucune paire (Prediction, MarketSnapshot) alignée trouvée avec ces filtres."))
            return

        trades = df_base.groupby(['symbol', 'model_name'], group_keys=False).apply(
            lambda g: compute_strategy_returns(g, use_log=use_log, cost_per_trade=cost)
        ).reset_index(drop=True)

        metrics = aggregate_metrics(trades, periods_per_year=ppy, rf_per_period=rf)

        # Export
        ts_tag = timezone.now().strftime("%Y%m%d_%H%M%S")
        outdir = os.path.join(outdir, ts_tag)
        os.makedirs(outdir, exist_ok=True)

        if fmt == "csv":
            trades.to_csv(os.path.join(outdir, "trades.csv"), index=False)
            metrics.to_csv(os.path.join(outdir, "metrics.csv"), index=False)
        else:
            trades.to_parquet(os.path.join(outdir, "trades.parquet"), index=False)
            metrics.to_parquet(os.path.join(outdir, "metrics.parquet"), index=False)

        # Affichage console rapide
        self.stdout.write(self.style.SUCCESS(f"OK - {len(trades)} lignes de trades, {len(metrics)} lignes de métriques."))
        self.stdout.write(self.style.SUCCESS(f"Dossier: {outdir}"))
        self.stdout.write("Aperçu metrics:")
        self.stdout.write(metrics.head(20).to_string(index=False))

    @staticmethod
    def _parse_dt(s: str):
        s = s.strip()
        if not s:
            return None
        try:
            # datetime sans timezone -> supposer time zone du projet
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if timezone.is_naive(dt):
                return timezone.make_aware(dt, timezone.get_current_timezone())
            return dt
        except Exception as e:
            raise CommandError(f"Format de date invalide: {s} (ISO attendu).")
