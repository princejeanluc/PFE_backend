from django.contrib import admin

# Register your models here.

from .models import Crypto, CryptoInfo, New

admin.site.register(Crypto)
admin.site.register(CryptoInfo)
admin.site.register(New)