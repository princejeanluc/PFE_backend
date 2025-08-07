import os
from datetime import datetime, timedelta
from django.utils.timezone import make_aware
import requests
import logging
from core.models import Crypto, CryptoInfo, MarketSnapshot, New
from django.db import transaction
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential



def get_hourly_sentiment_score(target_time: datetime) -> float:
    """
    Calcule le score de sentiment médian des titres publiés pendant l’heure de référence.
    Analyse via Azure Text Analytics.
    Retourne un score ∈ [-1, 1].
    """

    # 1️⃣ Fenêtre horaire
    start_time = make_aware(target_time.replace(minute=0, second=0, microsecond=0))
    end_time = start_time + timedelta(hours=1)

    # 2️⃣ Récupérer les titres
    titles = list(
        New.objects.filter(datetime__gte=start_time, datetime__lt=end_time)
        .values_list("title", flat=True)
    )

    if not titles:
        logging.warning(f"[SENTIMENT] Aucun article pour {start_time}")
        return 0.0

    # 3️⃣ Chunking si > 5000 caractères
    docs = []
    current_doc = ""
    for title in titles:
        if len(current_doc) + len(title) < 4900:
            current_doc += f"{title}. "
        else:
            docs.append(current_doc)
            current_doc = f"{title}. "
    if current_doc:
        docs.append(current_doc)

    # 4️⃣ Connexion Azure
    endpoint = os.getenv("AZURE_TEXT_API_ENDPOINT")
    key = os.getenv("AZURE_TEXT_API_KEY")
    client = TextAnalyticsClient(endpoint=endpoint, credential=AzureKeyCredential(key))

    try:
        response = client.analyze_sentiment(documents=docs, language="en")
        
        positive_scores, negative_scores = [], []
        for doc in response:
            if not doc.is_error:
                positive_scores.append(doc.confidence_scores.positive)
                negative_scores.append(doc.confidence_scores.negative)
            else:
                logging.error(f"[SENTIMENT] Erreur sur doc {doc.id}: {doc.error}")

        if not positive_scores:
            return 0.0
        
        # 5️⃣ Calcul score global
        avg_positive = sum(positive_scores) / len(positive_scores)
        avg_negative = sum(negative_scores) / len(negative_scores)
        score = avg_positive - avg_negative

        logging.info(f"[SENTIMENT] Score {score:.3f} ({len(docs)} batchs)")
        return score

    except Exception as e:
        logging.error(f"[SENTIMENT] Erreur Azure: {e}")
        print(f"[SENTIMENT] Erreur Azure: {e}")
        return 0.0
    





def save_market_snapshots(target_time: datetime, sentiment_score: float):
    """
    Crée un MarketSnapshot pour chaque crypto à l'heure donnée.
    - Utilise le dernier CryptoInfo disponible
    - Calcule le rendement horaire si possible
    - Affecte le sentiment_score
    """

    snapshot_time = make_aware(target_time.replace(minute=0, second=0, microsecond=0))
    one_hour_ago = snapshot_time - timedelta(hours=1)

    cryptos = Crypto.objects.all()
    created_count = 0

    with transaction.atomic():
        for crypto in cryptos:
            # Vérifier si déjà existant pour éviter doublons
            if MarketSnapshot.objects.filter(crypto=crypto, timestamp=snapshot_time).exists():
                logging.info(f"[MARKET SNAPSHOT] Snapshot déjà existant pour {crypto.symbol} à {snapshot_time}")
                continue

            # Dernière info dispo
            latest_info = (
                CryptoInfo.objects
                .filter(crypto=crypto, timestamp__lte=snapshot_time)
                .order_by("-timestamp")
                .first()
            )
            if not latest_info:
                logging.warning(f"[MARKET SNAPSHOT] Pas de CryptoInfo trouvé pour {crypto.symbol}")
                continue

            # Info précédente (pour rendement)
            prev_info = (
                CryptoInfo.objects
                .filter(crypto=crypto, timestamp__lte=one_hour_ago)
                .order_by("-timestamp")
                .first()
            )

            if prev_info and prev_info.current_price:
                hourly_return = (
                    (latest_info.current_price - prev_info.current_price)
                    / prev_info.current_price
                )
            else:
                hourly_return = 0.0

            # Création snapshot
            MarketSnapshot.objects.create(
                crypto=crypto,
                timestamp=snapshot_time,
                price=latest_info.current_price,
                volume=latest_info.total_volume or 0,
                hourly_return=hourly_return,
                news_positive_votes=0,
                news_negative_votes=0,
                sentiment_score=sentiment_score,
                predicted_frequency=1
            )
            created_count += 1

    logging.info(f"[MARKET SNAPSHOT] {created_count} snapshots créés pour {snapshot_time}")
    return created_count


def make_market_snapshot():
    now = datetime.now()
    score = get_hourly_sentiment_score(now)
    created = save_market_snapshots(now, score)
    print(f"{created} MarketSnapshots créés")