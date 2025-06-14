from rest_framework import routers
from .views import CryptoCurrencyViewSet, CryptoInfoViewSet, NewsArticleViewSet

router = routers.DefaultRouter()
router.register(r'cryptos', CryptoCurrencyViewSet, basename='cryptos')
router.register(r'infos', CryptoInfoViewSet, basename='infos')
router.register(r'news', NewsArticleViewSet, basename='news')
