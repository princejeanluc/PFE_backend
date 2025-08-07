import logging
import requests
import joblib
import numpy as np
from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput
import json
from dotenv import load_dotenv
import os
from datetime import timedelta
from core.models import Crypto, MarketSnapshot, Prediction
load_dotenv()

TRITON_SERVER_IP = os.getenv("VPS_IP")
TRITON_HTTP = f"http://{TRITON_SERVER_IP}:8000"
TRITON_GRPC = f"{TRITON_SERVER_IP}:8001"
MODEL_VERSION = "1"

# Cache du scaler pour éviter de le recharger à chaque appel
SCALER = joblib.load("scalers/scaler.pkl")
FEATURE_COLS = json.load(open("metadata/features.json", "r"))

triton_client = InferenceServerClient(url=TRITON_GRPC)

def predict_for_crypto(crypto_symbol, X_seq, X_base, current_price):
    """
    Effectue la prédiction GRU et XGBoost pour une crypto donnée.
    Retourne un dict avec log_return et prix prévu.
    """
    results = {}
    
    # === 1. GRU ===
    gru_model = f"gru_{crypto_symbol}"
    
    # Charger modèle
    load_url = f"{TRITON_HTTP}/v2/repository/models/{gru_model}/load"
    resp = requests.post(load_url)
    if resp.status_code != 200:
        print(f"Erreur chargement GRU {crypto_symbol}: {resp.text}")
    else:
        # Récupérer config du modèle
        config = requests.get(f"{TRITON_HTTP}/v2/models/{gru_model}/versions/{MODEL_VERSION}").json()
        input_name = config["inputs"][0]["name"]
        output_name = config["outputs"][0]["name"]
        
        # Créer input et output
        infer_input = InferInput(input_name, X_seq.shape, "FP32")
        infer_input.set_data_from_numpy(X_seq.astype(np.float32))
        infer_output = InferRequestedOutput(output_name)
        
        response = triton_client.infer(model_name=gru_model, model_version=MODEL_VERSION,
                                       inputs=[infer_input], outputs=[infer_output])
        pred_log_return_scaled = response.as_numpy(output_name)
        
        # Inverse transform
        n_features = SCALER.scale_.shape[0]
        target_index = FEATURE_COLS.index(f"return_{crypto_symbol}")
        full_vector = np.zeros((1, n_features))
        full_vector[:, target_index] = pred_log_return_scaled.flatten()
        pred_log_return = SCALER.inverse_transform(full_vector)[:, target_index][0]
        
        # Prix prévu
        predicted_price = current_price * np.exp(pred_log_return / 100)
        
        results["gru"] = {
            "log_return": float(pred_log_return),
            "predicted_price": float(predicted_price)
        }
        
        # Décharger modèle
        requests.post(f"{TRITON_HTTP}/v2/repository/models/{gru_model}/unload")
    
    # === 2. XGBoost ===
    xgb_model = f"xgboost_{crypto_symbol}"
    resp = requests.post(f"{TRITON_HTTP}/v2/repository/models/{xgb_model}/load")
    if resp.status_code != 200:
        print(f"Erreur chargement XGBoost {crypto_symbol}: {resp.text}")
    else:
        config = requests.get(f"{TRITON_HTTP}/v2/models/{xgb_model}/versions/{MODEL_VERSION}").json()
        input_name = config["inputs"][0]["name"]
        output_name = config["outputs"][0]["name"]
        
        infer_input = InferInput(input_name, X_base.shape, "FP32")
        infer_input.set_data_from_numpy(X_base.astype(np.float32))
        infer_output = InferRequestedOutput(output_name)
        
        response = triton_client.infer(model_name=xgb_model, model_version=MODEL_VERSION,
                                       inputs=[infer_input], outputs=[infer_output])
        pred_log_return = response.as_numpy(output_name).flatten()[0]
        predicted_price = current_price * np.exp(pred_log_return / 100)
        
        results["xgboost"] = {
            "log_return": float(pred_log_return),
            "predicted_price": float(predicted_price)
        }
        
        requests.post(f"{TRITON_HTTP}/v2/repository/models/{xgb_model}/unload")
    
    return results


def predict_all_cryptos():
    """
    Prédictions GRU et XGBoost uniquement pour les cryptos présentes dans features.json.
    """
    try:
        # 1. Charger les noms des colonnes
        with open("metadata/features.json", "r") as f:
            feature_cols = json.load(f)
        
        # 2. Extraire les symboles de crypto à partir des colonnes "return_{symbol}"
        crypto_symbols = set()
        for col in feature_cols:
            if col.startswith("return_"):
                symbol = col.replace("return_", "").lower()
                crypto_symbols.add(symbol)

        # 3. Récupérer les objets Crypto correspondants
        cryptos = Crypto.objects.filter(symbol__in=crypto_symbols)

        horizon = 48  # nombre de snapshots requis pour GRU

        for crypto in cryptos:
            try:
                # 4. Récupérer les derniers snapshots
                snapshots = MarketSnapshot.objects.filter(crypto=crypto).order_by('-timestamp')[:horizon]
                if len(snapshots) < horizon:
                    logging.warning(f"[PREDICT] Pas assez de données pour {crypto.symbol}")
                    continue

                snapshots = list(snapshots)[::-1]  # ordre chronologique croissant

                # 5. Construire les features
                df = {
                    f"return_{crypto.symbol.lower()}": [s.hourly_return for s in snapshots],
                    f"volume_{crypto.symbol.lower()}": [s.volume for s in snapshots],
                    "sentiment_market": [s.sentiment_score or 0.0 for s in snapshots]
                }
                X_df = np.array([df[c] for c in df]).T  # shape = (48, 3)

                # Séquentiel (GRU)
                X_seq = X_df[np.newaxis, :, :].astype(np.float32)

                # Statique (XGBoost)
                X_base = X_df[-1:, :].astype(np.float32)

                # 6. Prédiction
                price = snapshots[-1].price
                result = predict_for_crypto(crypto.symbol.lower(), X_seq, X_base, price)

                # 7. Enregistrement
                latest_snapshot = snapshots[-1]
                for model_name, model_result in result.items():
                    Prediction.objects.create(
                        crypto=crypto,
                        market_snapshot=latest_snapshot,
                        model_name=model_name,
                        predicted_log_return=model_result["log_return"],
                        predicted_price=model_result["predicted_price"],
                        predicted_date=latest_snapshot.timestamp + timedelta(hours=1),
                    )
                    logging.info(f"[PREDICT] {model_name.upper()} {crypto.symbol.upper()} → {model_result['predicted_price']:.2f}")

            except Exception as e:
                logging.error(f"[PREDICT] Erreur pour {crypto.symbol}: {e}")

    except Exception as e:
        logging.critical(f"[PREDICT] Échec initial: {e}")