# core.models.py
from django.db import models
from django.contrib.auth.models import AbstractUser
from django.db import models

class PosaUser(AbstractUser):
    credits = models.IntegerField(default=100)  # crédit initial
    type = models.CharField(max_length=100) # student , trader , etc ...
    has_accepted_terms = models.BooleanField(default=False)


class UserAction(models.Model):
    user = models.ForeignKey(PosaUser, on_delete=models.CASCADE)
    action_type = models.CharField(max_length=100)  # "simulation", "report", etc.
    cost_in_credits = models.IntegerField()
    timestamp = models.DateTimeField(auto_now_add=True)

class Crypto(models.Model):
    id = models.CharField(primary_key=True, max_length=100)
    name = models.CharField(max_length=100)
    symbol = models.CharField(max_length=10)
    image_url = models.URLField(null=True, blank=True, max_length=500)
    slug = models.SlugField(max_length=100, unique=True)

    def __str__(self):
        return self.name


class CryptoInfo(models.Model):
    crypto = models.ForeignKey(Crypto, on_delete=models.CASCADE, related_name='infos')
    timestamp = models.DateTimeField(null=True, blank=True)

    current_price = models.FloatField(null=True, blank=True)
    market_cap = models.BigIntegerField(null=True, blank=True)
    market_cap_rank = models.PositiveIntegerField(null=True, blank=True)
    fully_diluted_valuation = models.BigIntegerField(null=True, blank=True)

    total_volume = models.BigIntegerField(null=True, blank=True)
    high_24h = models.FloatField(null=True, blank=True)
    low_24h = models.FloatField(null=True, blank=True)
    price_change_24h = models.FloatField(null=True, blank=True)
    price_change_percentage_24h = models.FloatField(null=True, blank=True)
    market_cap_change_24h = models.BigIntegerField(null=True, blank=True)
    market_cap_change_percentage_24h = models.FloatField(null=True, blank=True)

    circulating_supply = models.FloatField(null=True, blank=True)
    total_supply = models.FloatField(null=True, blank=True)
    max_supply = models.FloatField(null=True, blank=True)

    ath = models.FloatField(null=True, blank=True)
    ath_change_percentage = models.FloatField(null=True, blank=True)
    ath_date = models.DateTimeField(null=True, blank=True)

    atl = models.FloatField(null=True, blank=True)
    atl_change_percentage = models.FloatField(null=True, blank=True)
    atl_date = models.DateTimeField(null=True, blank=True)

    last_updated = models.DateTimeField(null=True, blank=True)

    # Champs de gestion du risque
    volatility_24h = models.FloatField(null=True, blank=True)
    drawdown_from_ath = models.FloatField(null=True, blank=True)
    drawup_from_atl = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"{self.crypto.symbol.upper()} @ {self.timestamp}"


class Source(models.Model):
    name = models.CharField(max_length=255)
    url = models.URLField(max_length=500)

    def __str__(self):
        return self.name

class New(models.Model):
    cryptos = models.ManyToManyField(Crypto, related_name="news")
    source = models.ForeignKey(Source, on_delete=models.SET_NULL, null=True, related_name='articles')  # "relate"
    title = models.CharField(max_length=1000)
    url = models.URLField(max_length=500)
    datetime = models.DateTimeField()
    summary = models.TextField(default="")
    sentiment_score = models.FloatField(null=True, blank=True)  # abandonné
    positive = models.IntegerField(default=0)
    negative = models.IntegerField(default=0)
    tags = models.CharField(max_length=255, blank=True, null=True)  # Tags sous forme de texte brut séparé par virgule (améliorable)

    def __str__(self):
        return self.title

class MarketSnapshot(models.Model):
    crypto = models.ForeignKey(Crypto, on_delete=models.CASCADE)
    timestamp = models.DateTimeField()  # moment de l’instantané
    
    # Données figées pour prédiction (extraites de Crypto à t - 1h)
    price = models.FloatField()
    volume = models.FloatField()
    hourly_return = models.FloatField()

    # Données issues des news à t - 1h (agrégées)
    news_positive_votes = models.IntegerField(default=0)
    news_negative_votes = models.IntegerField(default=0)
    
    # Optionnel pour version future : polarité NLP (non utilisée pour l’instant)
    sentiment_score = models.FloatField(null=True, blank=True)
    
    # Résultat de la prédiction (fréquence attendue / étiquette)
    predicted_frequency = models.FloatField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)


class Portfolio(models.Model):
    user = models.ForeignKey(PosaUser, on_delete=models.CASCADE, related_name="portfolios")
    name = models.CharField(max_length=255)
    creation_date = models.DateTimeField(auto_now_add=True)
    holding_start = models.DateField()  # début période de détention
    holding_end = models.DateField()    # fin période de détention
    initial_budget = models.FloatField()

    ALLOCATION_TYPE_CHOICES = [
        ('manual', 'Manuelle'),
        ('autom', 'Automatique')
    ]
    allocation_type = models.CharField(max_length=20, choices=ALLOCATION_TYPE_CHOICES)
    objective = models.CharField(
        max_length=100,
        choices=[
            ("max_return", "Maximiser le rendement"),
            ("min_volatility", "Minimiser la volatilité"),
            ("sharpe", "Maximiser le ratio de Sharpe")
        ],
        blank=True,
        null=True
    )
    is_active = models.BooleanField(default=True)  # devient False une fois terminé

    def __str__(self):
        return f"{self.user.username} - {self.name}"



class PortfolioPerformance(models.Model):
    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name="performances")
    timestamp = models.DateTimeField()
    value = models.FloatField()  # valeur actuelle du portefeuille
    cumulative_return = models.FloatField()  # rendement cumulé
    volatility = models.FloatField(null=True, blank=True)
    sharpe_ratio = models.FloatField(null=True, blank=True)
    drawdown = models.FloatField(null=True, blank=True)
    sortino_ratio = models.FloatField(null=True, blank=True)
    expected_shortfall = models.FloatField(null=True, blank=True)
    value_at_risk = models.FloatField(null=True, blank=True)
    information_ratio = models.FloatField(null=True, blank=True)
    class Meta:
        ordering = ['timestamp']

class Holding(models.Model):
    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name='holdings')
    crypto = models.ForeignKey(Crypto, on_delete=models.CASCADE)
    allocation_percentage = models.FloatField()  # % du portefeuille
    quantity = models.FloatField()               # nombre d’unités détenues
    purchase_price = models.FloatField(null=True, blank=True)  # prix d’achat moyen
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.portfolio.name} - {self.crypto.symbol} ({self.allocation_percentage}%)"


class Prediction(models.Model):
    crypto = models.ForeignKey(Crypto, on_delete=models.CASCADE)
    market_snapshot = models.ForeignKey(MarketSnapshot, on_delete=models.CASCADE, related_name="predictions")
    model_name = models.CharField(max_length=100, default="default_model_v1")
    predicted_log_return = models.FloatField()
    predicted_price = models.FloatField(null=True, blank=True)  # optionnel, dérivé
    predicted_date = models.DateTimeField()
    predicted_volatility = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Prediction for {self.crypto.name} at {self.predicted_date}"
    
class ExogeneousVariables(models.Model): 
    crypto = models.ForeignKey(Crypto, on_delete=models.CASCADE)
    value = models.FloatField(null=True, blank=True)
    name = models.CharField(max_length=100, default="interest")
    timestamp = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Exogeneous variable {self.name} for  {self.crypto.name} at {self.timestamp}"
    


