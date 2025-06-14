from django.contrib import admin
from django.urls import path, include
from core.router import router  # ou depuis ton app API

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include(router.urls)),
]
