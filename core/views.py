# views.py
from contextlib import contextmanager
from datetime import timedelta
from threading import Thread
from rest_framework import viewsets, permissions, generics, filters
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.pagination import PageNumberPagination
from core.business.market.manager import MarketInfoManager
from core.business.risk.pricing import OptionPricingParams, price_option_mc
from core.business.simulation.allocation import create_performance_series, initialize_quantities, run_markowitz_allocation
from core.lib.policy import ASSISTANT_POLICY
from core.paginations import CryptoPagination
from .models import Crypto, CryptoInfo, MarketSnapshot, Portfolio, Holding, New, Prediction, StressScenario
from .serializers import (
    CryptoSerializer, CryptoInfoSerializer, CryptoTopSerializer, CryptoWithLatestInfoSerializer, MarketSnapshotSerializer, OptionPricingInputSerializer,
    PortfolioSerializer, HoldingSerializer,
    NewSerializer, PortfolioWithHoldingsSerializer, PosaUserSerializer, PredictionSerializer, RegisterSerializer, StressApplySerializer, StressScenarioSerializer
)
from django.contrib.auth import get_user_model



from allauth.socialaccount.providers.google.views import GoogleOAuth2Adapter
from dj_rest_auth.registration.views import SocialLoginView
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from allauth.socialaccount.models import SocialApp
from allauth.socialaccount.providers.oauth2.client import OAuth2Client
from dj_rest_auth.registration.views import SocialLoginView
from allauth.socialaccount.helpers import complete_social_login
from allauth.socialaccount.providers.google.provider import GoogleProvider


from allauth.socialaccount.adapter import get_adapter
from allauth.socialaccount.models import SocialLogin
from allauth.socialaccount.providers.google.views import GoogleOAuth2Adapter
from rest_framework_simplejwt.tokens import RefreshToken
from django.utils.timezone import now
import requests
from scipy.stats import spearmanr
from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np
import io
import pandas as pd
User = get_user_model()
import umap
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from core.business.risk.simulate import (
    RiskSimParams, to_hourly_median, log_returns_pct,
    simulate_with_ngarch, compute_metrics_from_paths, build_history_for_response
)
from core.business.risk.stress import apply_stress_to_portfolio
from core.constants import PREDICTION_MODELS
from django.db.models import OuterRef, Subquery

class RegisterView(generics.CreateAPIView):
    queryset = get_user_model().objects.all()
    serializer_class = RegisterSerializer
    permission_classes = [permissions.AllowAny]

    def create(self, request, *args, **kwargs):
        response = super().create(request, *args, **kwargs)
        user = self.get_queryset().get(id=response.data['id'])
        refresh = RefreshToken.for_user(user)
        return Response({
            "user": response.data,
            "refresh": str(refresh),
            "access": str(refresh.access_token)
        }, status=status.HTTP_201_CREATED)


def _safe_key(model_name: str) -> str:
    # clé d’annotation sûre (ex: "XGBoost" -> "xgboost")
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in model_name)

class CryptoViewSet(viewsets.ModelViewSet):
    queryset = Crypto.objects.all()
    serializer_class = CryptoSerializer
    pagination_class = CryptoPagination
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    def get_queryset(self):
        qs = Crypto.objects.all()

        # Pour chaque modèle, on annote la dernière prédiction (ordre: predicted_date, puis created_at)
        for model_name in PREDICTION_MODELS:
            key = _safe_key(model_name)
            base_sq = (
                Prediction.objects
                .filter(crypto=OuterRef('pk'), model_name=model_name)
                .order_by('-predicted_date', '-created_at')
            )

            qs = qs.annotate(
                **{
                    f'{key}_predicted_price':       Subquery(base_sq.values('predicted_price')[:1]),
                    f'{key}_predicted_log_return':  Subquery(base_sq.values('predicted_log_return')[:1]),
                    f'{key}_predicted_volatility':  Subquery(base_sq.values('predicted_volatility')[:1]),
                    f'{key}_predicted_date':        Subquery(base_sq.values('predicted_date')[:1]),
                    f'{key}_prediction_created_at': Subquery(base_sq.values('created_at')[:1]),
                }
            )

        return qs


class CryptoInfoViewSet(viewsets.ModelViewSet):
    queryset = CryptoInfo.objects.all()
    serializer_class = CryptoInfoSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    @action(detail=False, methods=['get'], url_path='top', permission_classes=[permissions.AllowAny])
    def get_top_cryptos(self, request):
        cryptos = Crypto.objects.all()
        latest = []

        for crypto in cryptos:
            info = crypto.infos.order_by("-timestamp").first()
            if not info or info.current_price is None:
                continue

            current_time = info.timestamp  # on prend la dernière dispo comme référence
            fallback_change_24h = None

            # Tentative de calcul manuel du rendement 24h
            earlier = crypto.infos.filter(
                timestamp__lte=current_time - timedelta(hours=24),
                current_price__isnull=False
            ).order_by("-timestamp").first()

            if earlier:
                try:
                    fallback_change_24h = 100 * (info.current_price - earlier.current_price) / earlier.current_price
                except ZeroDivisionError:
                    fallback_change_24h = None

            price_change = info.price_change_percentage_24h if info.price_change_percentage_24h is not None else fallback_change_24h

            latest.append({
                "id": crypto.id,
                "symbol": crypto.symbol,
                "name": crypto.name,
                "image_url": crypto.image_url,
                "current_price": info.current_price,
                "price_change_24h": round(price_change, 3) if price_change is not None else None,
                "market_cap": info.market_cap or 0,
            })
        latest.sort(key=lambda x: x["price_change_24h"] or -999, reverse=True)
        # Renvoyer top 10 par exemple
        return Response(CryptoTopSerializer(latest[:5], many=True).data)


class NewViewSet(viewsets.ModelViewSet):
    queryset = New.objects.all()
    serializer_class = NewSerializer
    permission_classes = [permissions.IsAuthenticated]

    @action(detail=False,methods=['get'], url_path='latest', permission_classes=[permissions.AllowAny])
    def latest_news(self,request):
        news = New.objects.order_by('-datetime')[:5]
        serialized = NewSerializer(news, many=True)
        return Response(serialized.data)
    
def _simulate_job(portfolio_id: int):
    cache.set(f"pf:{portfolio_id}:status", "running", 3600)
    from core.models import Portfolio
    portfolio = Portfolio.objects.get(pk=portfolio_id)
    try:
        if portfolio.allocation_type == "autom":
            run_markowitz_allocation(portfolio)

        if not portfolio.holdings.exists():
            cache.set(f"pf:{portfolio_id}:status", "error: no allocation", 600)
            return

        initialize_quantities(portfolio)
        create_performance_series(portfolio)

        cache.set(f"pf:{portfolio_id}:status", "ready", 3600)
    except Exception as e:
        cache.set(f"pf:{portfolio_id}:status", f"error: {e}", 600)

class PortfolioViewSet(viewsets.ModelViewSet):
    serializer_class = PortfolioSerializer
    permission_classes = [permissions.IsAuthenticated]
    filter_backends = [filters.SearchFilter]
    search_fields = ['name', 'holding_start','holding_end']

    def get_queryset(self):
        qs = Portfolio.objects.filter(user=self.request.user)

        name = self.request.query_params.get("name")
        crypto = self.request.query_params.get("crypto")
        start = self.request.query_params.get("start")
        end = self.request.query_params.get("end")

        if name:
            qs = qs.filter(name__icontains=name)
        if crypto:
            qs = qs.filter(holdings__crypto__symbol__iexact=crypto)
        if start and end:
            qs = qs.filter(holding_start__gte=start, holding_end__lte=end)
        return qs.distinct()
    def perform_create(self, serializer):
        serializer.save(user=self.request.user)
    
    def get_serializer_class(self):
        if self.request.method == "GET":
            return PortfolioWithHoldingsSerializer
        return PortfolioSerializer
    
    @action(detail=True, methods=['post'], url_path='simulate', permission_classes=[permissions.IsAuthenticated])
    def simulate_portfolio(self, request, pk=None):
        portfolio = self.get_queryset().get(pk=pk)
        # lance le job et répond tout de suite
        Thread(target=_simulate_job, args=(portfolio.id,), daemon=True).start()
        return Response({"detail": "Simulation démarrée", "status": "running"}, status=202)
    
    @action(detail=True, methods=["get"], url_path="crypto-returns", permission_classes=[permissions.IsAuthenticated])
    def crypto_returns(self, request, pk=None):
        portfolio = self.get_queryset().get(pk=pk)
        holdings = portfolio.holdings.all()

        start_date = portfolio.holding_start
        end_date = min(portfolio.holding_end, now().date())

        result = []

        for holding in holdings:
            infos = CryptoInfo.objects.filter(
                crypto=holding.crypto,
                timestamp__date__gte=start_date,
                timestamp__date__lte=end_date,
                current_price__isnull=False
            ).order_by("timestamp")

            if not infos.exists():
                continue

            df = pd.DataFrame([
                {"timestamp": info.timestamp, "price": info.current_price}
                for info in infos
            ])

            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
            df = df.resample("1h").median().dropna()

            if len(df) < 2:
                continue

            ret = (df["price"].iloc[-1] - df["price"].iloc[0]) / df["price"].iloc[0]

            result.append({
                "symbol": holding.crypto.symbol,
                "returns": [
                    {"timestamp": ts.isoformat(), "price": float(price)}
                    for ts, price in df["price"].items()
                ],
                "cumulative_return": round(ret, 6)
            })

        return Response(result)



class HoldingViewSet(viewsets.ModelViewSet):
    serializer_class = HoldingSerializer
    permission_classes = [permissions.IsAuthenticated]
    def get_queryset(self):
        return Holding.objects.filter(portfolio__user=self.request.user)



class PosaUserViewSet(viewsets.ModelViewSet):
    queryset = get_user_model().objects.all()
    serializer_class = PosaUserSerializer
    permission_classes = [permissions.IsAdminUser]  # ou change selon ta politique

class PredictionViewSet(viewsets.ModelViewSet):
    queryset = Prediction.objects.all()
    serializer_class = PredictionSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

class MarketSnapshotViewSet(viewsets.ModelViewSet):
    queryset = MarketSnapshot.objects.all()
    serializer_class = MarketSnapshotSerializer
    permission_classes = [permissions.AllowAny]

class GoogleAuthTokenView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        access_token = request.data.get("access_token")
        if not access_token:
            return Response({"error": "Access token is required."}, status=status.HTTP_400_BAD_REQUEST)

        # Obtenir les infos utilisateur depuis Google
        google_user_info_url = "https://www.googleapis.com/oauth2/v3/userinfo"
        headers = {"Authorization": f"Bearer {access_token}"}
        google_response = requests.get(google_user_info_url, headers=headers)

        if google_response.status_code != 200:
            return Response({"error": "Invalid Google token."}, status=status.HTTP_400_BAD_REQUEST)

        google_data = google_response.json()
        email = google_data.get("email")
        first_name = google_data.get("given_name")
        last_name = google_data.get("family_name")

        if not email:
            return Response({"error": "Email not available in Google data."}, status=status.HTTP_400_BAD_REQUEST)

        # Création ou récupération utilisateur
        user, created = User.objects.get_or_create(email=email, defaults={
            "username": email,
            "first_name": first_name or "",
            "last_name": last_name or "",
        })

        refresh = RefreshToken.for_user(user)
        return Response({
            "refresh": str(refresh),
            "access": str(refresh.access_token),
            "user": {
                "id": user.id,
                "email": user.email,
                "username": user.username,
                "name": f"{user.first_name} {user.last_name}".strip(),
            }
        })

class CurrentUserView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        serializer = PosaUserSerializer(request.user)
        return Response(serializer.data)
    

class MarketIndicatorsView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        manager = MarketInfoManager()
        data = manager.get_all_indicators()
        return Response(data)
    
class CryptoHistoryView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        symbol = request.GET.get("symbol")
        range = request.GET.get("range", "7d")  # Ex: 1d, 7d, 30d
        print(f"range = {range}")
        print(f"symbol{symbol}")
        if not symbol:
            return Response({"error": "Missing crypto_id"}, status=400)

        try:
            crypto = Crypto.objects.get(symbol= symbol)
        except Crypto.DoesNotExist:
            return Response({"error": "Crypto not found"}, status=404)

        # Convertir la période
        days = int(range.replace("d", "")) if "d" in range else 7
        start_date = now() - timedelta(days=days)

        history = CryptoInfo.objects.filter(
            crypto=crypto, timestamp__gte=start_date
        ).order_by("timestamp")

        response = {
            "crypto": {
                "id": crypto.id,
                "name": crypto.name,
                "symbol": crypto.symbol,
            },
            "current_price": history.last().current_price if history.exists() else None,
            "price_change_percentage_24h": history.last().price_change_percentage_24h if history.exists() else None,
            "history": [
                {"timestamp": h.timestamp, "price": h.current_price}
                for h in history
            ]
        }
        return Response(response)
    
# views.py
from django.db.models import F
from django.core.cache import cache
from django.utils.timezone import now
from rest_framework import permissions
from rest_framework.response import Response
from rest_framework.views import APIView
from datetime import timedelta

import numpy as np
import pandas as pd

# si tu as statsmodels
from statsmodels.tsa.stattools import grangercausalitytests

# ⚠️ utilise MarketSnapshot (déjà horaire) au lieu de CryptoInfo
from core.models import MarketSnapshot  # adapte l'import

class CryptoRelationMatrixView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        analysis_type = request.GET.get("type", "spearman")  # "spearman" | "granger"
        period = request.GET.get("period", "30d")            # "14d"
        lag = int(request.GET.get("lag", 1))                 # Granger uniquement

        # bornes de sécurité
        days = max(1, min(int(period.replace("d", "")), 180))  # cap à 180j
        since = now() - timedelta(days=days)

        # downsample optionnel si période très longue
        # (ici: garde 1 point sur 2 si > 90j)
        step = 2 if days > 90 else 1

        # limite le nombre de colonnes pour Granger (coût O(k²))
        top_k = int(request.GET.get("k", 20))  # par défaut 20, override via ?k=
        top_k = max(2, min(top_k, 50))         # garde des bornes raisonnables

        # ---------- CACHE ----------
        cache_key = f"relmat:v2:{analysis_type}:{days}:{lag}:{top_k}"
        cached = cache.get(cache_key)
        if cached:
            return Response(cached)

        # ---------- UNIQUE REQUÊTE ORM ----------
        # MarketSnapshot est déjà horaire ⇒ pas de resample.
        # On récupère (timestamp, symbol, price) pour la fenêtre.
        qs = (MarketSnapshot.objects
              .filter(timestamp__gte=since)
              .values_list("timestamp", "crypto__symbol", "price"))

        rows = list(qs)
        if not rows:
            return Response({"error": "Pas de données dans la période demandée."}, status=400)

        df = pd.DataFrame(rows, columns=["timestamp", "symbol", "price"])
        # Index horaire global (floor au cas où)
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.floor("h")

        # PIVOT: index = heure, colonnes = symbol, valeurs = price
        # MarketSnapshot étant horaire, pas besoin d'agg; sinon aggfunc="median"
        wide = df.pivot_table(index="timestamp", columns="symbol", values="price", aggfunc="median")

        # downsample si long
        if step > 1:
            wide = wide.iloc[::step, :]

        # drop colonnes trop vides
        keep_thresh = int(len(wide) * 0.7)
        wide = wide.dropna(axis=1, thresh=keep_thresh)

        if wide.shape[1] < 2:
            return Response({"error": "Pas assez de données pour établir des relations."}, status=400)

        # Normalise l’ensemble temporel (optionnel): on peut forward-fill sur petites lacunes
        wide = wide.sort_index().ffill(limit=2)

        # Pour Granger: sélectionne les top_k colonnes les plus “denses” et variables
        if analysis_type == "granger" and wide.shape[1] > top_k:
            # score = couverture non-null * variance
            coverage = wide.notna().sum() / len(wide)
            variance = wide.var(ddof=0).fillna(0)
            score = coverage * variance
            cols = score.sort_values(ascending=False).head(top_k).index
            wide = wide[cols]

        # ---------- CALCULS ----------
        if analysis_type == "spearman":
            # min_periods pour éviter les corr bruyantes
            matrix = wide.corr(method="spearman", min_periods=max(12, lag+2))

        elif analysis_type == "granger":
            # on travaille sur rendements (stationnarité)
            returns = np.log(wide).diff().dropna()
            symbols = list(returns.columns)
            k = len(symbols)
            M = pd.DataFrame(np.zeros((k, k)), index=symbols, columns=symbols)

            # Optim: on pré-filtre les paires quasi colinéaires ou trop lacunaires
            valid_len = (returns.notna().sum() > (lag + 8))
            symbols = [s for s in symbols if valid_len[s]]
            if len(symbols) < 2:
                return Response({"error": "Données insuffisantes après nettoyage."}, status=400)

            # Recalibre la matrice avec ce sous-ensemble
            M = pd.DataFrame(np.zeros((len(symbols), len(symbols))), index=symbols, columns=symbols)

            # Boucle double (k²), mais bornée par top_k
            # Utilise try/except léger et séries sans NaN
            for cause in symbols:
                for effect in symbols:
                    if cause == effect:
                        continue
                    pair = returns[[effect, cause]].dropna()
                    if len(pair) <= lag + 8:
                        continue
                    try:
                        # verbose=False évite l'overhead I/O
                        res = grangercausalitytests(pair, maxlag=lag, verbose=False)
                        p = res[lag][0]["ssr_ftest"][1]
                        M.loc[cause, effect] = round(1 - float(p), 3)
                    except Exception:
                        # garde 0 en cas d’échec
                        pass
            matrix = M
        else:
            return Response({"error": "Type d'analyse non supporté"}, status=400)

        out = {
            "type": analysis_type,
            "period": f"{days}d",
            "lag": lag,
            "matrix": matrix.to_dict(),
            "cryptos": list(matrix.columns)
        }

        # ---------- CACHE (15 min spearman / 30 min granger) ----------
        timeout = 15 * 60 if analysis_type == "spearman" else 30 * 60
        cache.set(cache_key, out, timeout=timeout)

        return Response(out)

    
class CryptoMapView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        # paramètres (bornés)
        days = int(request.GET.get("days", 30))
        days = max(1, min(days, 180))
        since = now() - timedelta(days=days)
        min_points = int(request.GET.get("min_points", 24))  # densité min par symbole
        top_k = int(request.GET.get("k", 200))               # limite nb de cryptos
        top_k = max(10, min(top_k, 500))

        cache_key = f"cryptomap:v3:{days}:{min_points}:{top_k}"
        cached = cache.get(cache_key)
        if cached:
            return Response(cached)

        # 1) UNIQUE REQUÊTE ORM (déjà horaire) → pas de resample
        qs = (MarketSnapshot.objects
              .filter(timestamp__gte=since,
                      price__isnull=False,
                      volume__isnull=False)
              .values_list("timestamp", "crypto__symbol", "price", "volume",
                           "crypto__image_url"))
        rows = list(qs)
        if not rows:
            return Response({"error": "Pas de données dans la période demandée."}, status=400)

        df = pd.DataFrame(rows, columns=["timestamp", "symbol", "price", "volume", "image"])
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.floor("H")
        df = df.sort_values(["symbol", "timestamp"])

        # 2) Filtre densité par symbole (évite count() par-crypto en DB)
        counts = df.groupby("symbol")["timestamp"].size()
        dense_symbols = counts[counts >= min_points].index
        if len(dense_symbols) < 2:
            return Response({"error": "Pas assez de données"}, status=400)
        # limite top_k plus “informatif” (variance x couverture)
        sub = df[df["symbol"].isin(dense_symbols)]

        # score simple pour limiter k si besoin
        # couverture ~ proportion d’heures distinctes
        hours_per_symbol = sub.groupby("symbol")["timestamp"].nunique()
        # variance de prix
        var_per_symbol = sub.groupby("symbol")["price"].var(ddof=0).fillna(0.0)
        score = (hours_per_symbol / hours_per_symbol.max()) * (var_per_symbol / (var_per_symbol.max() or 1.0))
        keep = score.sort_values(ascending=False).head(top_k).index
        sub = sub[sub["symbol"].isin(keep)]

        # 3) Agrégations vectorisées (first/last/std/mean) par symbole
        agg_idx = sub.groupby("symbol")["timestamp"].agg(first="first", last="last")
        price_agg = sub.groupby("symbol")["price"].agg(first="first", last="last", std="std", mean="mean")
        vol_agg   = sub.groupby("symbol")["volume"].agg(first="first", last="last", mean="mean")
        img_map = sub.groupby("symbol")["image"].first()

        # métriques
        # return = (last - first) / first (avec garde-fous sur division)
        eps = 1e-12
        ret = (price_agg["last"] - price_agg["first"]) / (np.abs(price_agg["first"]) + eps)
        vol_price = price_agg["std"].fillna(0.0)
        vol_change = (vol_agg["last"] - vol_agg["first"]) / (np.abs(vol_agg["first"]) + eps)
        avg_volume = vol_agg["mean"].fillna(0.0)

        df_features = pd.DataFrame({
            "symbol": ret.index,
            "return": ret.values.astype(float),
            "volatility": vol_price.values.astype(float),
            "volume_change": vol_change.values.astype(float),
            "avg_volume": avg_volume.values.astype(float),
        }).reset_index(drop=True)

        if df_features.shape[0] < 2:
            return Response({"error": "Pas assez de données"}, status=400)

        # 4) Normalisation + UMAP + DBSCAN
        X = df_features[["return", "volatility", "volume_change", "avg_volume"]].to_numpy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # UMAP: 2D, voisins par défaut OK (données agrégées)
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        embedding = reducer.fit_transform(X_scaled)

        clustering = DBSCAN(eps=0.7, min_samples=3).fit(embedding)
        labels = clustering.labels_

        # 5) Format de sortie (évite .iloc dans boucle → index alignés)
        sym = df_features["symbol"].tolist()
        out = [{
            "symbol": s,
            "image": img_map.get(s),
            "x": float(embedding[i, 0]),
            "y": float(embedding[i, 1]),
            "cluster": (int(labels[i]) if labels[i] != -1 else None),
            "metrics": {
                "return": float(df_features.at[i, "return"]),
                "volatility": float(df_features.at[i, "volatility"]),
                "volume_change": float(df_features.at[i, "volume_change"]),
                "avg_volume": float(df_features.at[i, "avg_volume"]),
            }
        } for i, s in enumerate(sym)]

        result = {"points": out}
        # cache 15 min (c’est un “map” exploratoire)
        cache.set(cache_key, result, 15 * 60)
        return Response(result)


class LatestCryptoInfoView(generics.ListAPIView):
    queryset = Crypto.objects.all()
    serializer_class = CryptoWithLatestInfoSerializer
    # permission_classes = [permissions.IsAuthenticated]
    filter_backends = [filters.SearchFilter]
    search_fields = ['name', 'symbol']



class RiskSimulationView(APIView):
    # permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        """
        Query params:
          - symbol: str (ex: BTC)
          - horizon_hours: int (<= 72 recommandé)
          - n_sims: int (ex: 200)
        """
        symbol = request.query_params.get("symbol")
        horizon_hours = int(request.query_params.get("horizon_hours", 72))
        n_sims = int(request.query_params.get("n_sims", 200))

        if not symbol:
            return Response({"error": "Missing symbol"}, status=400)

        # 1) Historique: 6 mois glissants, prix horaires (médiane)
        since = now() - timedelta(days=180)

        try:
            crypto = Crypto.objects.get(symbol=symbol)
        except Crypto.DoesNotExist:
            return Response({"error": "Crypto not found"}, status=404)

        qs = CryptoInfo.objects.filter(
            crypto=crypto, timestamp__gte=since, current_price__isnull=False
        ).order_by("timestamp")

        if not qs.exists():
            return Response({"error": "No data for this crypto"}, status=404)

        df = pd.DataFrame([{"timestamp": r.timestamp, "price": r.current_price} for r in qs])
        df_hourly = to_hourly_median(df)  # 1H median + dropna

        if len(df_hourly) < 50:
            return Response({"error": "Not enough hourly data"}, status=400)

        # 2) log-return (%) = ln(Pt/Pt-1)*100
        lr_pct = log_returns_pct(df_hourly["price"])
        last_price = float(df_hourly["price"].iloc[-1])

        # 3) NGARCH + loi mixte → simulate (à brancher dans simulate_with_ngarch)
        try:
            paths, vol = simulate_with_ngarch(
                logret_pct=lr_pct.values,
                last_price=last_price,
                horizon_hours=horizon_hours,
                n_sims=n_sims
            )
        except Exception as e:
            raise e
            return Response({"error": f"Simulation failed: {e}"}, status=500)

        # 4) métriques sur distribution terminale
        metrics = compute_metrics_from_paths(paths)

        # 5) format de réponse
        t0 = df_hourly.index[-1]
        forecast_index = pd.date_range(t0 + pd.Timedelta(hours=1), periods=horizon_hours, freq="1h")

        history_payload = build_history_for_response(df_hourly, keep_last_hours=24*3)

        resp = {
            "symbol": crypto.symbol,
            "history": history_payload,
            "forecast_timestamps": [ts.isoformat() for ts in forecast_index.to_pydatetime()],
            "paths": paths.tolist(),    # n_sims x T
            "vol": [float(v) for v in vol],   # T
            "metrics": metrics
        }
        return Response(resp, status=200)

class OptionPricingView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        ser = OptionPricingInputSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        payload = ser.validated_data

        # 1) construire horizon_hours si dates fournies
        horizon_hours = payload.get("horizon_hours")
        if horizon_hours is None:
            cur = payload["current_date"]
            mat = payload["maturity_date"]
            delta = mat - cur
            horizon_hours = max(1, int(delta.total_seconds() // 3600))

        # 2) récupérer 6 mois d'historique (médiane horaire)
        try:
            crypto = Crypto.objects.get(symbol=payload["symbol"])
        except Crypto.DoesNotExist:
            return Response({"error": "Crypto not found"}, status=404)

        since = now() - timedelta(days=180)
        qs = CryptoInfo.objects.filter(
            crypto=crypto,
            timestamp__gte=since,
            current_price__isnull=False
        ).order_by("timestamp")

        if not qs.exists():
            return Response({"error": "No data for this crypto"}, status=404)

        df = pd.DataFrame([{"timestamp": r.timestamp, "price": r.current_price} for r in qs])
        df_hourly = to_hourly_median(df)  # 1H median + dropna
        if len(df_hourly) < 50:
            return Response({"error": "Not enough hourly data"}, status=400)

        # 3) paramètres & pricing
        params = OptionPricingParams(
            symbol=payload["symbol"],
            option_type=payload["option_type"].lower(),
            strike=float(payload["strike"]),
            risk_free=float(payload.get("risk_free", 0.0)),
            horizon_hours=int(horizon_hours),
            n_sims=int(payload.get("n_sims", 1000)),
        )

        # Cap de sécurité côté vue aussi
        params.n_sims = min(max(params.n_sims, 100), 2000)
        params.horizon_hours = min(max(params.horizon_hours, 1), 24*7)

        try:
            result = price_option_mc(df_hourly, params)
        except Exception as e:
            # Message lisible côté front (et tu peux journaliser le détail côté serveur)
            return Response({"error": f"Pricing failed: {str(e)}"}, status=503)

        return Response({
            "symbol": params.symbol,
            "option_type": params.option_type,
            "strike": params.strike,
            "risk_free": params.risk_free,
            "horizon_hours": params.horizon_hours,
            "n_sims": params.n_sims,
            "price": result["price"],
            "ci95": result["ci95"],
            "stderr": result["stderr"],
            "diagnostics": {
                "model_used": result["model_used"],
                "last_price": result["last_price"],
            }
        }, status=200)
    

class StressScenarioListView(generics.ListAPIView):
    queryset = StressScenario.objects.filter(is_active=True)
    serializer_class = StressScenarioSerializer


class StressApplyView(generics.GenericAPIView):
    serializer_class = StressApplySerializer

    def post(self, request, *args, **kwargs):
        ser = self.get_serializer(data=request.data)
        ser.is_valid(raise_exception=True)
        data = ser.validated_data

        # Charger le scénario
        if "id" in data["scenario"]:
            sc = StressScenario.objects.get(id=data["scenario"]["id"])
            scenario = StressScenarioSerializer(sc).data
        else:
            scenario = data["scenario"]

        # Charger le portefeuille
        portfolio = Portfolio.objects.get(id=data["portfolio_id"])

        # Appliquer le stress
        result = apply_stress_to_portfolio(portfolio, scenario)
        return Response(result, status=status.HTTP_200_OK)
    



# views_llm.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import permissions
from core.permissions import IsLLMRequest
from urllib.parse import urlparse
class LLMListPortfolios(APIView):
    permission_classes = [IsLLMRequest]
    def get(self, request):
        qs = Portfolio.objects.filter(user=request.user).only("id","name").order_by("-creation_date")[:20]
        return Response([{"id": p.id, "name": p.name} for p in qs])

class LLMPortfolioSummary(APIView):
    permission_classes = [IsLLMRequest]
    def get(self, request, pk: int):
        pf = Portfolio.objects.filter(user=request.user, pk=pk).first()
        if not pf:
            return Response({"error": "Not found"}, status=404)
        ser = PortfolioWithHoldingsSerializer(pf)  # déjà prêt chez toi :contentReference[oaicite:4]{index=4}
        # Option : “aplatir”/réduire ici pour un payload minimal
        return Response(ser.data)



# views_assist.py (ou dans ton views.py)
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.conf import settings
import os, asyncio, jwt
from django.utils.timezone import datetime
from fastmcp import Client as MCPClient
from google import genai
from google.genai import types as gtypes
import tempfile

@contextmanager
def temp_environ(updates: dict):
    """Injecte des variables d'env le temps d'un bloc, puis restaure."""
    old = {k: os.environ.get(k) for k in updates}
    os.environ.update({k: v for k, v in updates.items() if v is not None})
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v

def _mint_tool_token(user_id: int, scope=("portfolio:read","news:read"), ttl_s=90) -> str:
    claims = {
        "sub": str(user_id),
        "aud": "llm",
        "scope": list(scope),
        "exp": datetime.now() + timedelta(seconds=ttl_s),
    }
    print(f"claims {claims}")
    print(f"secret{settings.SECRET_KEY}")
    return jwt.encode(claims, settings.SECRET_KEY, algorithm="HS256")

class AssistBriefView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        # par défaut: 7 jours / 50 titres / FR
        since_hours = int(request.data.get("since_hours", 168))
        limit       = int(request.data.get("limit", 50))
        lang        = request.data.get("lang", "fr")
        risk        = request.data.get("risk_profile", "prudent")

        tool_token = _mint_tool_token(request.user.id)
        model_key  = os.getenv("LLM_MODEL_KEY", "CHANGE_ME")

        SYSTEM = f"""
Tu es un analyste crypto rigoureux.
Tu n'as accès qu'à des **titres + liens + sources** (aucun contenu d'article).
Procède ainsi :
1) Appelle **recent_article_titles** avec (since_hours={since_hours}, limit={limit}, lang="{lang}").
2) À partir des TITRES uniquement, regroupe par thèmes (réglementaire, ETF, DeFi, L2, hacks, stablecoins, macro…).
3) Produis un brief **Markdown** :
   # Brief marché (dernières {since_hours//24}j)
   ## Thèmes clés (avec 1–2 bullets chacun)
   ## Liste des articles (titre + [lien]) — max {limit}
   ## Conseils prudents (3 max) — basés sur les tendances des TITRES uniquement
Règles :
- Pas d'invention : tu ne spécules pas au-delà de ce que la formulation des TITRES suggère.
- Mentionne la **source** (nom du média si disponible) et le **lien**.
- Français, concis, pas de promesses.
"""

        USER_MSG = (
            f"Analyse les actualités crypto des {since_hours//24} derniers jours à partir des TITRES "
            f"(max {limit}, lang={lang}) et propose 3 conseils prudents adaptés à un profil {risk}. "
            f"Retourne uniquement du Markdown."
        )

        async def _run():
            mcp = MCPClient("mcp_server.py")  # Chemin vers le serveur MCP ci-dessus
            gem = genai.Client(api_key=os.getenv("GOOGLE_AI_API_KEY"))
            async with mcp:
                # Initialiser l'auth côté MCP (hors LLM)
                await mcp.call_tool("_auth_set", {"tool_token": tool_token, "model_key": model_key})

                # Réponse Markdown (tools activés)
                resp = await gem.aio.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=USER_MSG,
                    config=gtypes.GenerateContentConfig(
                        system_instruction=SYSTEM,
                        tools=[mcp.session],
                        temperature=0.2,
                        max_output_tokens=1800,
                    ),
                )
                return (resp.text or "").strip()

        markdown = asyncio.run(_run())
        return Response({"markdown": markdown})
    


class AssistChatView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        msg   = (request.data.get("message") or "").strip()
        hist  = request.data.get("history") or []  # liste de textes libres (optionnel)
        lang  = (request.data.get("lang") or "fr").strip()
        risk  = (request.data.get("risk_profile") or "prudent").strip()

        if not msg:
            return Response({"markdown": "_(Écris un message pour démarrer la discussion.)_"}, status=200)

        tool_token = _mint_tool_token(request.user.id)
        model_key  = os.getenv("LLM_MODEL_KEY", "CHANGE_ME")

        # Contexte utilisateur minimal (sans s’alourdir)
        context = f"(Langue: {lang} | Profil de risque: {risk})"
        history_text = ""
        if hist:
            # Concaténation simple pour rester léger (tu pourras passer à un vrai historique multi-tour plus tard)
            history_text = "\n".join(f"- {t}" for t in hist[-8:])  # garde max 8 tours

        USER_MSG = (
            f"{context}\n"
            f"Historique (facultatif):\n{history_text}\n\n"
            f"Question de l’utilisateur:\n{msg}\n\n"
            f"Réponds en Markdown. Utilise des outils MCP si nécessaire."
        )

        async def _run():
            mcp = MCPClient("mcp_server.py")   # ← ton serveur MCP (avec _auth_set, recent_article_titles, etc.)
            gem = genai.Client(api_key=os.getenv("GOOGLE_AI_API_KEY"))
            async with mcp:
                # Initialiser l’auth côté MCP (hors LLM)
                await mcp.call_tool("_auth_set", {"tool_token": tool_token, "model_key": model_key})

                # LLM avec tools génériques
                resp = await gem.aio.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=USER_MSG,
                    config=gtypes.GenerateContentConfig(
                        system_instruction=ASSISTANT_POLICY,
                        tools=[mcp.session],
                        temperature=0.2,
                        max_output_tokens=1400,
                    ),
                )
                return (resp.text or "").strip()

        markdown = asyncio.run(_run())
        return Response({"markdown": markdown})