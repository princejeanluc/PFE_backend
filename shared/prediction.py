import logging
import requests
import joblib
import numpy as np
import json
from dotenv import load_dotenv
import os
from datetime import timedelta
from core.models import Crypto, MarketSnapshot, Prediction

load_dotenv()

TRITON_SERVER_IP = os.getenv("VPS_IP")
TRITON_HTTP = f"http://{TRITON_SERVER_IP}:8000"
MODEL_VERSION = "1"

SCALER = joblib.load("shared/scalers/scaler.pkl")
FEATURE_COLS = json.load(open("shared/metadata/features.json", "r"))

def infer_http(model_name: str, input_data: np.ndarray) -> np.ndarray:
    """
    Envoie une requête HTTP à Triton Server pour faire une inférence.
    """
    url = f"{TRITON_HTTP}/v2/models/{model_name}/versions/{MODEL_VERSION}/infer"

    infer_input = {
        "name": "input__0",
        "shape": list(input_data.shape),
        "datatype": "FP32",
        "data": input_data.flatten().tolist()
    }

    payload = {
        "inputs": [infer_input],
        "outputs": [{"name": "output__0"}]
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()
    result = response.json()

    data = result["outputs"][0]["data"]
    return np.array(data, dtype=np.float32)

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

def predict_all_cryptos():
    """
    Lance les prédictions sur toutes les cryptos disponibles dans le fichier features.json.
    """
    try:
        with open("shared/metadata/features.json", "r") as f:
            feature_cols = json.load(f)

        crypto_symbols = set()
        for col in feature_cols:
            if col.startswith("return_"):
                symbol = col.replace("return_", "").lower()
                crypto_symbols.add(symbol)

        cryptos = Crypto.objects.filter(symbol__in=crypto_symbols)
        horizon = 48

        for crypto in cryptos:
            try:
                snapshots = MarketSnapshot.objects.filter(crypto=crypto).order_by('-timestamp')[:horizon]
                print("len snapshots = ",len(snapshots))
                if len(snapshots) < horizon:
                    logging.warning(f"[PREDICT] Pas assez de données pour {crypto.symbol}")
                    continue

                snapshots = list(snapshots)[::-1]
                df = {
                    f"return_{crypto.symbol.lower()}": [s.hourly_return for s in snapshots],
                    f"volume_{crypto.symbol.lower()}": [s.volume for s in snapshots],
                    "sentiment_market": [s.sentiment_score or 0.0 for s in snapshots]
                }

                X_df = np.array([df[c] for c in df]).T
                X_seq = X_df[np.newaxis, :, :].astype(np.float32)
                X_base = X_df[-1:, :].astype(np.float32)

                current_price = snapshots[-1].price
                predictions = predict_for_crypto(crypto.symbol.lower(), X_seq, X_base, current_price)

                for model_name, model_result in predictions.items():
                    Prediction.objects.create(
                        crypto=crypto,
                        market_snapshot=snapshots[-1],
                        model_name=model_name,
                        predicted_log_return=model_result["log_return"],
                        predicted_price=model_result["predicted_price"],
                        predicted_date=snapshots[-1].timestamp + timedelta(hours=1)
                    )
                    logging.info(f"[PREDICT] {model_name.upper()} {crypto.symbol.upper()} → {model_result['predicted_price']:.2f}")

            except Exception as e:
                logging.error(f"[PREDICT] Erreur {crypto.symbol}: {e}")

    except Exception as e:
        logging.critical(f"[PREDICT] Échec initial: {e}")
