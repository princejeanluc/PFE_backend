from rest_framework import serializers
from .models import Crypto, CryptoInfo, New, Source

class CryptoSerializer(serializers.ModelSerializer):
    class Meta:
        model = Crypto
        fields = '__all__'

class CryptoInfoSerializer(serializers.ModelSerializer):
    class Meta:
        model = CryptoInfo
        fields = '__all__'

class SourceSerializer(serializers.ModelSerializer):
    class Meta:
        model = Source
        fields = '__all__'

class NewsSerializer(serializers.ModelSerializer):
    class Meta:
        model = New
        fields = '__all__'
