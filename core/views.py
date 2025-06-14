from django.shortcuts import render

from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework.decorators import action
from .models import Crypto, CryptoInfo, New
from .serializers import CryptoSerializer, CryptoInfoSerializer, NewsSerializer

class CryptoViewSet(viewsets.ModelViewSet):
    queryset = Crypto.objects.all()
    serializer_class = CryptoSerializer

    @action(detail=True, methods=['get'])
    def info(self, request, pk=None):
        crypto = self.get_object()
        infos = CryptoInfo.objects.filter(crypto=crypto).order_by('-timestamp')
        serializer = CryptoInfoSerializer(infos, many=True)
        return Response(serializer.data)

class CryptoInfoViewSet(viewsets.ModelViewSet):
    queryset = CryptoInfo.objects.all()
    serializer_class = CryptoInfoSerializer

class NewsViewSet(viewsets.ModelViewSet):
    queryset = New.objects.all().order_by('-datetime')
    serializer_class = NewsSerializer

    @action(detail=False, methods=['get'])
    def by_crypto(self, request):
        crypto_id = request.query_params.get('crypto')
        if crypto_id:
            articles = self.queryset.filter(crypto__id=crypto_id)
        else:
            articles = self.queryset
        serializer = NewsSerializer(articles, many=True)
        return Response(serializer.data)

