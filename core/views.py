from django.shortcuts import render

from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework.decorators import action
from .models import CryptoCurrency, CryptoInfo, NewsArticle
from .serializers import CryptoCurrencySerializer, CryptoInfoSerializer, NewsArticleSerializer

class CryptoCurrencyViewSet(viewsets.ModelViewSet):
    queryset = CryptoCurrency.objects.all()
    serializer_class = CryptoCurrencySerializer

    @action(detail=True, methods=['get'])
    def info(self, request, pk=None):
        crypto = self.get_object()
        infos = CryptoInfo.objects.filter(crypto=crypto).order_by('-timestamp')
        serializer = CryptoInfoSerializer(infos, many=True)
        return Response(serializer.data)

class CryptoInfoViewSet(viewsets.ModelViewSet):
    queryset = CryptoInfo.objects.all()
    serializer_class = CryptoInfoSerializer

class NewsArticleViewSet(viewsets.ModelViewSet):
    queryset = NewsArticle.objects.all().order_by('-published_at')
    serializer_class = NewsArticleSerializer

    @action(detail=False, methods=['get'])
    def by_crypto(self, request):
        crypto_id = request.query_params.get('crypto')
        if crypto_id:
            articles = self.queryset.filter(crypto__id=crypto_id)
        else:
            articles = self.queryset
        serializer = NewsArticleSerializer(articles, many=True)
        return Response(serializer.data)

