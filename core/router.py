from rest_framework import routers
from .views import CryptoViewSet, CryptoInfoViewSet, NewsViewSet

router = routers.DefaultRouter()
router.register(r'cryptos', CryptoViewSet, basename='cryptos')
router.register(r'infos', CryptoInfoViewSet, basename='infos')
router.register(r'news', NewsViewSet, basename='news')
