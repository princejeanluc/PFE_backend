# serializers.py
from rest_framework import serializers
from django.contrib.auth import get_user_model
from .models import Crypto, CryptoInfo, New, PortfolioPerformance, Prediction, Portfolio, Holding, MarketSnapshot

# Utilisateur personnalisé
User = get_user_model()


class RegisterSerializer(serializers.ModelSerializer):
    class Meta:
        model = get_user_model()
        fields = ['id', 'email', 'username', 'password', 'type']
        extra_kwargs = {'password': {'write_only': True}}

    def create(self, validated_data):
        user = get_user_model().objects.create_user(**validated_data)
        return user
class PosaUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name', 'type', 'credits', 'date_joined']

# Crypto de base
class CryptoSerializer(serializers.ModelSerializer):
    class Meta:
        model = Crypto
        fields = ['id', 'symbol', 'name', 'slug', 'image_url']

# Info temporelle sur une crypto
class CryptoInfoSerializer(serializers.ModelSerializer):
    class Meta:
        model = CryptoInfo
        fields = ['id', 'crypto', 'timestamp', 'price', 'volume', 'market_cap']

# News + votes (tu avais dit que pos/neg sont plus importants que sentiments)
class NewSerializer(serializers.ModelSerializer):
    cryptos = CryptoSerializer(many=True, read_only=True)
    class Meta:
        model = New
        fields = ['id', 'cryptos', 'title', 'summary', 'source', 'url', 'datetime']

# Prédictions (résultats du modèle)
class PredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prediction
        fields = ['id', 'crypto', 'timestamp', 'predicted_price', 'predicted_return', 'volatility']

# Portefeuille
class PortfolioSerializer(serializers.ModelSerializer):
    # user = serializers.PrimaryKeyRelatedField(queryset=User.objects.all())

    class Meta:
        model = Portfolio
        fields = [
            'id','user', 'name', 'creation_date',
            'holding_start', 'holding_end', 'initial_budget',
            'allocation_type', 'objective', 'is_active'
        ]
        read_only_fields = ['user']

# Composition du portefeuille
class HoldingSerializer(serializers.ModelSerializer):
    crypto = serializers.SlugRelatedField(
        slug_field='id',
        queryset=Crypto.objects.all()
    )
    crypto_detail = CryptoSerializer(source='crypto', read_only=True)
    class Meta:
        model = Holding
        fields = [
            'id', 'portfolio', 'crypto','crypto_detail',
            'allocation_percentage', 'quantity', 'purchase_price', 'created_at'
        ]

class MarketSnapshotSerializer(serializers.ModelSerializer):
    class Meta:
        model = MarketSnapshot
        fields = ['id', 'timestamp', 'crypto', 'price', 'volume', 'market_cap']

# serializers.py
class CryptoWithLatestInfoSerializer(serializers.ModelSerializer):
    latest_info = serializers.SerializerMethodField()

    class Meta:
        model = Crypto
        fields = ['id', 'symbol', 'name', 'slug', 'image_url', 'latest_info']

    def get_latest_info(self, obj):
        latest = obj.infos.order_by('-timestamp').first()
        if latest:
            return {
                "timestamp": latest.timestamp,
                "current_price": latest.current_price,
                "market_cap": latest.market_cap,
                "market_cap_rank": latest.market_cap_rank,
                "volatility_24h": latest.volatility_24h,
            }
        return None
    


class PortfolioPerformanceSerializer(serializers.ModelSerializer):
    class Meta:
        model = PortfolioPerformance
        fields = ["id", "timestamp","value", "cumulative_return","volatility","sharpe_ratio","drawdown", "sortino_ratio", "expected_shortfall", "value_at_risk","information_ratio" ]


class PortfolioWithHoldingsSerializer(serializers.ModelSerializer):
    holdings = HoldingSerializer(many=True, read_only=True)
    performances = PortfolioPerformanceSerializer(many=True, read_only=True)
    
    class Meta:
        model = Portfolio
        fields = [
            'id', 'name', 'creation_date',
            'holding_start', 'holding_end', 'initial_budget',
            'allocation_type', 'objective',
            'is_active', 'holdings', 'performances'
        ]


class CryptoTopSerializer(serializers.Serializer):
    id = serializers.CharField()
    symbol = serializers.CharField()
    name = serializers.CharField()
    image_url = serializers.URLField()
    current_price = serializers.FloatField()
    price_change_24h = serializers.FloatField()
    market_cap = serializers.IntegerField()
