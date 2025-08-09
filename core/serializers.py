# serializers.py
from rest_framework import serializers
from django.contrib.auth import get_user_model
from .models import Crypto, CryptoInfo, New, PortfolioPerformance, Prediction, Portfolio, Holding, MarketSnapshot, StressScenario
from django.db.models import Max
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
    latest_predictions = serializers.SerializerMethodField()

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
                "return_24h":latest.price_change_percentage_24h,
                "circulating_supply":latest.circulating_supply
            }
        return None
    
    def get_latest_predictions(self, obj):
        # 1) Dernière date de prédiction pour CETTE crypto
        latest_date = (obj.prediction_set
                         .aggregate(Max('predicted_date'))
                         .get('predicted_date__max'))
        if not latest_date:
            return []

        # 2) Prédictions à cette date (optionnel: limiter aux 2 modèles connus)
        qs = (obj.prediction_set
                .filter(predicted_date=latest_date)
                # .filter(model_name__in=['XGBoost', 'GRU'])  # <- décommente si tu veux figer à ces 2 modèles
                .only('model_name','predicted_price','predicted_log_return',
                      'predicted_volatility','predicted_date','created_at')
                .order_by('model_name','-created_at'))

        # 3) Garder la plus récente par modèle (réduction côté Python, portable)
        by_model = {}
        for p in qs:
            if p.model_name not in by_model:
                by_model[p.model_name] = {
                    "model_name": p.model_name,
                    "predicted_price": p.predicted_price,
                    "predicted_log_return": p.predicted_log_return,
                    "predicted_volatility": p.predicted_volatility,
                    "predicted_date": p.predicted_date,
                    "created_at": p.created_at,
                }

        # 4) Retourner une liste (ex. deux entrées: XGBoost, GRU)
        # tri optionnel, pour l’affichage
        return sorted(by_model.values(), key=lambda x: x["model_name"].lower())
    


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

