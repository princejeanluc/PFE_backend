# serializers.py
from rest_framework import serializers
from django.contrib.auth import get_user_model
from .models import Crypto, CryptoInfo, New, PortfolioPerformance, Prediction, Portfolio, Holding, MarketSnapshot, StressScenario
from core.constants import PREDICTION_MODELS
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


def _safe_key(model_name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in model_name)

# serializers.py
class CryptoWithLatestInfoSerializer(serializers.ModelSerializer):
    latest_info = serializers.SerializerMethodField()
    latest_predictions = serializers.SerializerMethodField()

    class Meta:
        model = Crypto
        fields = ['id', 'symbol', 'name', 'slug', 'image_url', 'latest_info', 'latest_predictions']

    def get_latest_info(self, obj):
        latest = obj.infos.order_by('-timestamp').first()
        if latest:
            return {
                "timestamp": latest.timestamp,
                "current_price": latest.current_price,
                "market_cap": latest.market_cap,
                "market_cap_rank": latest.market_cap_rank,
                "volatility_24h": latest.volatility_24h,
                "return_24h":latest.price_change_percentage_24h,
                "circulating_supply":latest.circulating_supply
            }
        return None
    
    def get_latest_predictions(self, obj):
        out = []

        # 1) Essaye d’abord les annotations (rapide, pas de N+1)
        annotated_found = False
        for model_name in PREDICTION_MODELS:
            key = _safe_key(model_name)
            price      = getattr(obj, f'{key}_predicted_price', None)
            log_ret    = getattr(obj, f'{key}_predicted_log_return', None)
            vol        = getattr(obj, f'{key}_predicted_volatility', None)
            pred_date  = getattr(obj, f'{key}_predicted_date', None)
            created_at = getattr(obj, f'{key}_prediction_created_at', None)

            if any(v is not None for v in (price, log_ret, vol, pred_date, created_at)):
                annotated_found = True
                out.append({
                    "model_name": model_name,
                    "predicted_price": price,
                    "predicted_log_return": log_ret,
                    "predicted_volatility": vol,
                    "predicted_date": pred_date,
                    "created_at": created_at,
                })

        if annotated_found:
            out.sort(key=lambda x: x["model_name"].lower())
            return out

        # 2) Fallback portable : dernier par modèle (sans DISTINCT ON), 1 mini-requête par modèle
        for model_name in PREDICTION_MODELS:
            pred = (obj.prediction_set
                    .filter(model_name=model_name)
                    .order_by('-predicted_date', '-created_at')
                    .only('model_name','predicted_price','predicted_log_return',
                          'predicted_volatility','predicted_date','created_at')
                    .first())
            if pred:
                out.append({
                    "model_name": pred.model_name,
                    "predicted_price": pred.predicted_price,
                    "predicted_log_return": pred.predicted_log_return,
                    "predicted_volatility": pred.predicted_volatility,
                    "predicted_date": pred.predicted_date,
                    "created_at": pred.created_at,
                })

        out.sort(key=lambda x: x["model_name"].lower())
        return out


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

class OptionPricingInputSerializer(serializers.Serializer):
    symbol = serializers.CharField()
    option_type = serializers.ChoiceField(choices=["call", "put"])
    strike = serializers.FloatField(min_value=0)
    risk_free = serializers.FloatField(required=False, default=0.0)  # annuel (ex 0.02)
    # soit horizon_hours, soit (current_date, maturity_date)
    horizon_hours = serializers.IntegerField(required=False, min_value=1, max_value=24*7)
    current_date = serializers.DateTimeField(required=False)
    maturity_date = serializers.DateTimeField(required=False)
    n_sims = serializers.IntegerField(required=False, min_value=100, max_value=2000, default=1000)

    def validate(self, attrs):
        # horizon_hours direct ?
        h = attrs.get("horizon_hours")
        cur = attrs.get("current_date")
        mat = attrs.get("maturity_date")
        if h is None:
            if cur is None or mat is None:
                raise serializers.ValidationError(
                    "Provide either 'horizon_hours' or both 'current_date' and 'maturity_date'."
                )
        return attrs

class StressScenarioSerializer(serializers.ModelSerializer):
    class Meta:
        model = StressScenario
        fields = '__all__'


class StressApplySerializer(serializers.Serializer):
    portfolio_id = serializers.IntegerField()
    scenario = serializers.JSONField()  # {"id": 1} ou inline complet

