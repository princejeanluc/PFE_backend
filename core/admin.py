from django.contrib import admin

# Register your models here.

from .models import Crypto, CryptoInfo, New, Source

admin.site.register(Crypto)
admin.site.register(CryptoInfo)
admin.site.register(New)
admin.site.register(Source)