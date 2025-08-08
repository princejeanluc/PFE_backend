from django.contrib import admin

# Register your models here.

from .models import Crypto, CryptoInfo, New, Source, Portfolio, PortfolioPerformance,Prediction, PosaUser,MarketIndicatorSnapshot,MarketSnapshot

admin.site.register(Crypto)
admin.site.register(CryptoInfo)
admin.site.register(New)
admin.site.register(Source)
admin.site.register(Portfolio)
admin.site.register(Prediction)
admin.site.register(PortfolioPerformance)
admin.site.register(PosaUser)
admin.site.register(MarketIndicatorSnapshot)
admin.site.register(MarketSnapshot)