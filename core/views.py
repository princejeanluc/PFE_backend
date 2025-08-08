# views.py
from datetime import timedelta
from rest_framework import viewsets, permissions, generics, filters
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.pagination import PageNumberPagination
from core.business.market.manager import MarketInfoManager
from core.business.simulation.allocation import create_performance_series, initialize_quantities, run_markowitz_allocation
from .models import Crypto, CryptoInfo, MarketSnapshot, Portfolio, Holding, New, Prediction
from .serializers import (
    CryptoSerializer, CryptoInfoSerializer, CryptoTopSerializer, CryptoWithLatestInfoSerializer, MarketSnapshotSerializer,
    PortfolioSerializer, HoldingSerializer,
    NewSerializer, PortfolioWithHoldingsSerializer, PosaUserSerializer, PredictionSerializer, RegisterSerializer
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
    
class CryptoViewSet(viewsets.ModelViewSet):
    queryset = Crypto.objects.all()
    serializer_class = CryptoSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]


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

        # 1. ALLOCATION
        if portfolio.allocation_type == "autom":
            run_markowitz_allocation(portfolio)

        if not portfolio.holdings.exists():
            return Response({"error": "Aucune allocation trouvée."}, status=400)

        # 2. INITIALISATION DES QUANTITÉS
        initialize_quantities(portfolio)

        # 3. SIMULATION
        create_performance_series(portfolio)

        return Response({"detail": "Simulation effectuée avec succès."}, status=200)
    
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
            df = df.resample("1H").median().dropna()

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
    
class CryptoRelationMatrixView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        analysis_type = request.GET.get("type", "spearman")  # "spearman" or "granger"
        period = request.GET.get("period", "30d")  # format: "14d"
        lag = int(request.GET.get("lag", 1))  # utilisé seulement pour Granger

        # Fenêtre temporelle
        days = int(period.replace("d", ""))
        since = now() - timedelta(days=days)

        cryptos = Crypto.objects.all()
        data = {}
        i = 0
        for crypto in cryptos:
            infos = CryptoInfo.objects.filter(
                crypto=crypto, timestamp__gte=since
            ).order_by("timestamp")

            if infos.exists():
                series = pd.Series(
                    [info.current_price for info in infos],
                    index=pd.to_datetime([info.timestamp for info in infos]),
                    name=crypto.symbol
                )
                # Agrégation par heure : médiane
                series_hourly = series.resample('1H').median()
                data[crypto.symbol] = series_hourly
                i += 1

        df = pd.DataFrame(data)

        # Supprimer uniquement les colonnes trop vides (ex: plus de 30% de NaN)
        df = df.dropna(axis=1, thresh=int(len(df) * 0.7))

        if df.empty or df.shape[1] < 2:
            print("Pas assez de données : df vide ou < 2 colonnes")
            return Response({"error": "Pas assez de données pour établir des relations."}, status=400)

        # Type d’analyse
        if analysis_type == "spearman":
            matrix = df.corr(method='spearman')

        elif analysis_type == "granger":
            symbols = df.columns
            matrix = pd.DataFrame(np.zeros((len(symbols), len(symbols))), columns=symbols, index=symbols)

            for cause in symbols:
                for effect in symbols:
                    if cause == effect:
                        matrix.loc[cause, effect] = 0
                        continue

                    test_df = df[[effect, cause]].dropna()
                    if len(test_df) > lag + 2:
                        try:
                            result = grangercausalitytests(test_df, maxlag=lag, verbose=False)
                            p_value = result[lag][0]['ssr_ftest'][1]
                            matrix.loc[cause, effect] = round(1 - p_value, 3)
                        except Exception:
                            matrix.loc[cause, effect] = 0
                    else:
                        matrix.loc[cause, effect] = 0
        else:
            return Response({"error": "Type d'analyse non supporté"}, status=400)
        dictMatrix = matrix.to_dict()
        return Response({
            "type": analysis_type,
            "period": period,
            "lag": lag,
            "matrix": dictMatrix, 
            "cryptos": dictMatrix.keys()
        })
    

class CryptoMapView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    def get(self, request):
        days = 30
        since = now() - timedelta(days=days)

        cryptos = Crypto.objects.all()
        features = []

        for crypto in cryptos:
            infos = CryptoInfo.objects.filter(
                crypto=crypto,
                timestamp__gte=since,
                current_price__isnull=False,
                total_volume__isnull=False
            ).order_by("timestamp")

            if infos.count() < 24:  # on s'assure qu'on a des données suffisantes
                continue

            df = pd.DataFrame.from_records([
                {
                    "timestamp": info.timestamp,
                    "price": info.current_price,
                    "volume": info.total_volume
                }
                for info in infos
            ])

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df = df.resample("1H").median().dropna()

            if df.empty or len(df) < 10:
                continue

            # Métriques à calculer
            ret = (df["price"].iloc[-1] - df["price"].iloc[0]) / df["price"].iloc[0]
            vol = df["price"].std()
            vol_change = (df["volume"].iloc[-1] - df["volume"].iloc[0]) / df["volume"].iloc[0]
            avg_volume = df["volume"].mean()

            features.append({
                "symbol": crypto.symbol,
                "image": crypto.image_url,
                "metrics": {
                    "return": ret,
                    "volatility": vol,
                    "volume_change": vol_change,
                    "avg_volume": avg_volume
                }
            })

        if len(features) < 2:
            return Response({"error": "Pas assez de données"}, status=400)

        # Construction du DataFrame
        df_features = pd.DataFrame([
            {
                "symbol": f["symbol"],
                "return": f["metrics"]["return"],
                "volatility": f["metrics"]["volatility"],
                "volume_change": f["metrics"]["volume_change"],
                "avg_volume": f["metrics"]["avg_volume"]
            }
            for f in features
        ])

        # Normalisation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_features.drop(columns=["symbol"]))

        # Réduction UMAP
        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding = reducer.fit_transform(X_scaled)

        # Clustering DBSCAN
        clustering = DBSCAN(eps=0.7, min_samples=3).fit(embedding)
        labels = clustering.labels_
        # Format final
        result = []
        for i, f in enumerate(features):
            result.append({
                "symbol": f["symbol"],
                "image": f["image"],
                "x": float(embedding[i][0]),
                "y": float(embedding[i][1]),
                "cluster": int(labels[i]) if labels[i] != -1 else None,  # -1 = outlier
                "metrics": f["metrics"]
            })
        return Response({"points": result})


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
        forecast_index = pd.date_range(t0 + pd.Timedelta(hours=1), periods=horizon_hours, freq="1H")

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
