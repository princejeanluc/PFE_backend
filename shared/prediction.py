import logging
import requests
import joblib
import numpy as np
import json
from dotenv import load_dotenv
import os
from datetime import timedelta
from core.models import Crypto, MarketSnapshot, Prediction
from django.utils.timezone import make_aware, now as tz_now
from django.utils import timezone
load_dotenv()

# Chargements “globaux”
FEATURES_PATH = "shared/metadata/features.json"
SCALER_PATH   = "shared/scalers/scaler.pkl"

with open(FEATURES_PATH, "r", encoding="utf-8") as f:
    FEATURE_COLS = json.load(f)

import joblib
SCALER = joblib.load(SCALER_PATH)

TRITON_SERVER_IP = os.getenv("VPS_IP")
TRITON_HTTP = f"http://{TRITON_SERVER_IP}:8000"
MODEL_VERSION = "1"
WINDOW = 48  # taille de fenêtre GRU




def ensure_aware_utc(dt):
    """Retourne dt en UTC aware, arrondi à l’heure pile."""
    # arrondir d'abord
    dt = dt.replace(minute=0, second=0, microsecond=0)
    if timezone.is_naive(dt):
        # si tu veux tout travailler en UTC:
        return timezone.make_aware(dt, timezone.utc)
    # sinon, convertir en UTC
    return dt.astimezone(timezone.utc)




def get_model_io(model_name: str):
    # Assure le chargement
    requests.post(f"{TRITON_HTTP}/v2/repository/models/{model_name}/load")

    r = requests.get(f"{TRITON_HTTP}/v2/models/{model_name}")
    r.raise_for_status()
    cfg = r.json()

    in0 = cfg["inputs"][0]
    out0 = cfg["outputs"][0]
    return {
        "input_name": in0["name"],
        "input_shape": in0["shape"],
        "input_dtype": in0.get("data_type", "FP32"),
        "output_name": out0["name"],
        "output_shape": out0["shape"],
        "output_dtype": out0.get("data_type", "FP32"),
    }






def infer_http(model_name: str, input_data: np.ndarray) -> np.ndarray:
    # 1) Lire la config pour savoir si une batch dim est requise
    io = get_model_io(model_name)  # -> dict avec: input_name, output_name, input_shape, dtype, max_batch_size
    inp_name = io["input_name"]
    out_name = io["output_name"]
    exp_shape = io["input_shape"]      # ex: [-1, 57] pour XGB, [-1, 48, 57] pour GRU
    dtype = io["input_dtype"]                # "FP32" etc.

    x = np.asarray(input_data, dtype=np.float32)

    # 2) Ajuster la forme selon la config
    #    - XGBoost: exp_shape ~ [-1, 57]  -> x doit être [B, 57]
    #    - GRU:     exp_shape ~ [-1, T, F] -> x doit être [B, T, F]
    if (exp_shape[0] == -1) and (len(x.shape) == len(exp_shape) - 1):
        # ajoute la batch dim devant
        x = x.reshape((1, *x.shape))
    # si max_batch_size == 0, on laisse la forme “sans batch dim”

    # 3) Construire le payload HTTP v2
    infer_input = {
        "name": inp_name,
        "shape": list(x.shape),
        "datatype": dtype,
        "data": x.flatten().tolist()
    }
    payload = {
        "inputs": [infer_input],
        "outputs": [{"name": out_name}]
    }

    url = f"{TRITON_HTTP}/v2/models/{model_name}/infer"
    res = requests.post(url, json=payload, timeout=15)
    if res.status_code != 200:
        raise RuntimeError(f"Infer {model_name} failed: {res.status_code} {res.text}")

    out = res.json()["outputs"][0]
    return np.asarray(out["data"], dtype=np.float32)




WINDOW = 48

def build_feature_tensor(feature_cols, end_time):
    """
    Construit X_seq (1, WINDOW, F) et X_base (F,) en respectant l'ordre des colonnes
    défini dans features.json, pour la fenêtre qui se termine à end_time (arrondie à l’heure).
    Hypothèse: MarketSnapshot contient une ligne par (crypto, timestamp) alignée sur l’heure.
    """
    from django.utils.timezone import make_aware
    # end_time = make_aware(end_time.replace(minute=0, second=0, microsecond=0)) # déjà make aware
    start_time = end_time - timedelta(hours=WINDOW)

    # Récupérons d’abord les symboles nécessaires
    symbols = set()
    for c in feature_cols:
        if c.startswith("return_") or c.startswith("volume_"):
            symbols.add(c.split("_", 1)[1].lower())

    # Précharger tous les snapshots utiles en 1 requête par symbole
    # et indexer par timestamp arrondi à l’heure
    by_symbol = {}
    for sym in symbols:
        qs = (MarketSnapshot.objects
              .filter(crypto__symbol__iexact=sym,
                      timestamp__gte=start_time, timestamp__lt=end_time + timedelta(hours=1))
              .order_by("timestamp"))
        by_symbol[sym] = list(qs)

    # Préparer la grille temporelle (48 timesteps)
    times = [start_time + timedelta(hours=i) for i in range(WINDOW)]

    # Construire la matrice [WINDOW, F]
    F = len(feature_cols)
    X_mat = np.zeros((WINDOW, F), dtype=np.float32)

    for t_idx, ts in enumerate(times):
        # sentiment du marché: on prend, p.ex., le median des sentiments disponibles,
        # sinon 0.0; à défaut, si tu stockes déjà un sentiment par snapshot, prends-le du symbole target
        # Ici on met 0.0 par défaut:
        sentiment_val = 0.0

        # petit cache par timestamp des snapshots (facultatif)
        snap_cache = {}

        for f_idx, col in enumerate(feature_cols):
            if col == "sentiment_market":
                X_mat[t_idx, f_idx] = sentiment_val
                continue

            kind, sym = col.split("_", 1)  # "return" ou "volume", puis symbole
            sym = sym.lower()

            if sym not in by_symbol:
                # donnée manquante -> 0.0
                continue

            if ts not in snap_cache:
                # trouver le snapshot à ts (exact) pour ce symbole
                # (si tes timestamps ne sont pas EXACTEMENT à l’heure pile,
                # fais un arrondi côté écriture)
                match = next((s for s in by_symbol[sym] if s.timestamp.replace(minute=0, second=0, microsecond=0) == ts), None)
                snap_cache[ts] = {}
                snap_cache[ts][sym] = match
            else:
                match = snap_cache[ts].get(sym)

            if not match:
                continue

            if kind == "return":
                X_mat[t_idx, f_idx] = float(match.hourly_return or 0.0)
            elif kind == "volume":
                X_mat[t_idx, f_idx] = float(match.volume or 0.0)

    # X_seq pour GRU
    X_seq = X_mat[np.newaxis, :, :]  # (1, WINDOW, F)
    # X_base pour XGBoost
    X_base = X_mat[-1, :]            # (F,)

    return X_seq, X_base



def predict_for_crypto(crypto_symbol, X_seq, X_base, current_price):
    """
    Effectue la prédiction GRU et XGBoost pour une crypto donnée via HTTP API Triton.
    """
    results = {}

    try:
        # Charger modèle GRU
        model_gru = f"gru_{crypto_symbol}"
        requests.post(f"{TRITON_HTTP}/v2/repository/models/{model_gru}/load")

        # Inférence
        pred_gru_scaled = infer_http(model_gru, X_seq)

        # Inverse scaling
        target_index = FEATURE_COLS.index(f"return_{crypto_symbol}")
        n_features = SCALER.scale_.shape[0]
        full_vector = np.zeros((1, n_features))
        full_vector[0, target_index] = pred_gru_scaled[0]
        pred_gru = SCALER.inverse_transform(full_vector)[0, target_index]
        predicted_price_gru = current_price * np.exp(pred_gru / 100)

        results["gru"] = {
            "log_return": float(pred_gru),
            "predicted_price": float(predicted_price_gru)
        }

        requests.post(f"{TRITON_HTTP}/v2/repository/models/{model_gru}/unload")

    except Exception as e:
        logging.warning(f"[GRU] Erreur {crypto_symbol}: {e}")

    try:
        # Charger modèle XGBoost
        model_xgb = f"xgboost_{crypto_symbol}"
        requests.post(f"{TRITON_HTTP}/v2/repository/models/{model_xgb}/load")

        pred_xgb = infer_http(model_xgb, X_base)[0]
        predicted_price_xgb = current_price * np.exp(pred_xgb / 100)

        results["xgboost"] = {
            "log_return": float(pred_xgb),
            "predicted_price": float(predicted_price_xgb)
        }

        requests.post(f"{TRITON_HTTP}/v2/repository/models/{model_xgb}/unload")

    except Exception as e:
        logging.warning(f"[XGBOOST] Erreur {crypto_symbol}: {e}")

    return results



def _unload_model_safely(model_name: str):
    """Décharge un modèle Triton sans casser si déjà déchargé."""
    try:
        requests.post(f"{TRITON_HTTP}/v2/repository/models/{model_name}/unload", timeout=5)
    except Exception as _:
        pass

def _extract_symbols_from_features(feature_cols):
    """Retourne l’ensemble des symboles présents sous forme return_<sym> / volume_<sym>."""
    symbols = set()
    for c in feature_cols:
        if c.startswith("return_") or c.startswith("volume_"):
            sym = c.split("_", 1)[1].lower()
            symbols.add(sym)
    return symbols

def predict_all_cryptos(end_time=None):
    """
    - Construit X_seq/X_base à partir de MarketSnapshot pour la fenêtre se terminant à end_time (arrondie à l’heure).
    - Prévoit via Triton (HTTP) pour GRU et XGBoost.
    - Enregistre en base (Prediction).
    - Décharge les modèles à la fin.
    """
    if end_time is None:
        end_time = timezone.now()  # déjà aware

    end_time = ensure_aware_utc(end_time)

    # 1) Déterminer les symboles couverts par les features d’entraînement
    covered_symbols = _extract_symbols_from_features(FEATURE_COLS)

    # 2) Limiter aux cryptos existantes en base & couvertes par l’entraînement
    cryptos = Crypto.objects.filter(symbol__in=covered_symbols)

    # 3) Indices utiles
    #    - index de la colonne cible (log-return) par crypto dans l’espace “entraîné”
    target_index_map = {}
    for sym in covered_symbols:
        target_col = f"return_{sym}"
        try:
            target_index_map[sym] = FEATURE_COLS.index(target_col)
        except ValueError:
            # Au cas où la colonne cible n’existe pas (ne devrait pas arriver si covered_symbols est issu de features.json)
            target_index_map[sym] = None

    created = 0
    skipped = 0

    for crypto in cryptos:
        sym = crypto.symbol.lower()
        # 4) Vérifier qu’on a bien l’index de cible
        t_idx = target_index_map.get(sym)
        if t_idx is None:
            logging.warning(f"[PREDICT] {sym}: colonne cible absente des features -> skip")
            skipped += 1
            continue

        # 5) Construire les tenseurs X_seq / X_base (ordre strict de FEATURE_COLS)
        try:
            X_seq, X_base = build_feature_tensor(FEATURE_COLS, end_time=end_time)
        except Exception as e:
            logging.warning(f"[PREDICT] {sym}: build_feature_tensor a échoué: {e}")
            skipped += 1
            continue

        # Vérifier la fenêtre et l’historique
        if X_seq.shape[1] != WINDOW or X_seq.shape[2] != len(FEATURE_COLS):
            logging.warning(f"[PREDICT] {sym}: shape inattendue de X_seq {X_seq.shape} (attendu (1,{WINDOW},{len(FEATURE_COLS)}))")
            skipped += 1
            continue

        # 6) Dernier snapshot pour récupérer le prix courant et timestamp
        #    On exige WINDOW snapshots alignés -> le dernier est la fin de fenêtre
        last_snap = (MarketSnapshot.objects
                     .filter(crypto=crypto, timestamp__lte=end_time)
                     .order_by('-timestamp')
                     .first())
        if not last_snap:
            logging.warning(f"[PREDICT] {sym}: aucun MarketSnapshot à <= {end_time}")
            skipped += 1
            continue

        current_price = float(last_snap.price)

        # ---------------------------
        #       PRÉDICTION GRU
        # ---------------------------
        try:
            gru_name = f"gru_{sym}"

            # (get_model_io charge le modèle et donne les bons noms/shape)
            _ = get_model_io(gru_name)  # charge + check I/O

            y_scaled = infer_http(gru_name, X_seq)  # retourne (batch, 1) ou (1,) selon le modèle
            # Assure qu’on a un scalaire
            y_scaled = float(np.array(y_scaled).flatten()[0])

            # Inverse scaling :
            n_features = SCALER.scale_.shape[0]
            vec = np.zeros((1, n_features), dtype=np.float32)
            vec[0, t_idx] = y_scaled
            y_unscaled = float(SCALER.inverse_transform(vec)[0, t_idx])

            price_gru = current_price * np.exp(y_unscaled / 100.0)

            Prediction.objects.create(
                crypto=crypto,
                market_snapshot=last_snap,
                model_name="gru",
                predicted_log_return=y_unscaled,
                predicted_price=price_gru,
                predicted_date=last_snap.timestamp + timedelta(hours=1)
            )
            created += 1
        except Exception as e:
            logging.warning(f"[GRU] {sym}: {e}", exc_info=True)
        finally:
            _unload_model_safely(gru_name)

        # ---------------------------
        #     PRÉDICTION XGBOOST
        # ---------------------------
        try:
            xgb_name = f"xgboost_{sym}"

            # charge + check I/O
            _ = get_model_io(xgb_name)

            # X_base attendu par FIL : souvent [F] (ou [1,F], géré par infer_http)
            y_xgb = infer_http(xgb_name, X_base)
            y_xgb = float(np.array(y_xgb).flatten()[0])  # XGBoost a été entraîné “brut” (pas de scaler)

            price_xgb = current_price * np.exp(y_xgb / 100.0)

            Prediction.objects.create(
                crypto=crypto,
                market_snapshot=last_snap,
                model_name="xgboost",
                predicted_log_return=y_xgb,
                predicted_price=price_xgb,
                predicted_date=last_snap.timestamp + timedelta(hours=1)
            )
            created += 1
        except Exception as e:
            logging.warning(f"[XGBOOST] {sym}: {e}", exc_info=True)
        finally:
            _unload_model_safely(xgb_name)

    logging.info(f"[PREDICT] done. created={created}, skipped={skipped}")
    return {"created": created, "skipped": skipped}