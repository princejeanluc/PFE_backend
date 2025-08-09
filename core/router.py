# core/router.py

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    CryptoHistoryView, CryptoMapView, CryptoRelationMatrixView, CryptoViewSet, CryptoInfoViewSet, CurrentUserView, GoogleAuthTokenView, LatestCryptoInfoView, MarketIndicatorsView,  MarketSnapshotViewSet, OptionPricingView,
    PortfolioViewSet, HoldingViewSet,
    NewViewSet, PosaUserViewSet, PredictionViewSet, RegisterView, RiskSimulationView, StressApplyView, StressScenarioListView
)
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)

router = DefaultRouter()
router.register(r'cryptos', CryptoViewSet, basename='crypto')
router.register(r'crypto-infos', CryptoInfoViewSet, basename='crypto-info')
router.register(r'portfolios', PortfolioViewSet, basename='portfolio')
router.register(r'holdings', HoldingViewSet, basename='holding')
router.register(r'news', NewViewSet, basename='news')
router.register(r'users', PosaUserViewSet, basename='user')
router.register(r'market-snapshots', MarketSnapshotViewSet, basename='market-snapshot')
router.register(r'predictions', PredictionViewSet, basename='prediction')

urlpatterns = [
    path('cryptos/latest-info/', LatestCryptoInfoView.as_view(), name='latest-crypto-info'),
    path('', include(router.urls)),
    # Get self user information
    path('auth/user/', CurrentUserView.as_view(), name='auth_user'),
    # JWT Auth
    path('auth/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('auth/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),

    # Register
    path('auth/register/', RegisterView.as_view(), name='auth_register'),

    # dj-rest-auth
    path('auth/', include('dj_rest_auth.urls')),
    path('auth/registration/', include('dj_rest_auth.registration.urls')),

    # Google OAuth2
    path("auth/google/token/", GoogleAuthTokenView.as_view(), name="google_token_login"),
]


urlpatterns += [
    path("market/indicators/", MarketIndicatorsView.as_view(), name="market-indicators"),
    path("market/history/", CryptoHistoryView.as_view(), name="crypto-history"),
    path("crypto-relations/", CryptoRelationMatrixView.as_view(), name="crypto-relations"),
    path("crypto-map/", CryptoMapView.as_view(), name="crypto-map"),
    path("risk/simulate/", RiskSimulationView.as_view(), name="risk-simulate"),
    path("risk/option/price/", OptionPricingView.as_view(), name="option-price"),
    path("risk/stress/scenarios/", StressScenarioListView.as_view(), name="stress-scenarios"),
    path("risk/stress/apply/", StressApplyView.as_view(), name="stress-apply"),


] 

