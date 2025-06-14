from django.db import models

class Crypto(models.Model):
    id = models.CharField(primary_key=True, max_length=100)
    name = models.CharField(max_length=100)
    symbol = models.CharField(max_length=10)
    image_url = models.URLField(null=True, blank=True)
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
    url = models.URLField()

    def __str__(self):
        return self.name


class New(models.Model):
    crypto = models.ForeignKey(Crypto, on_delete=models.CASCADE, related_name='news_items')  # Association: "Link"
    source = models.ForeignKey(Source, on_delete=models.SET_NULL, null=True, related_name='articles')  # "relate"
    title = models.CharField(max_length=255)
    url = models.URLField()
    datetime = models.DateTimeField()
    summary = models.TextField()
    sentiment_score = models.FloatField(null=True, blank=True)  # Nouveau champ
    tags = models.CharField(max_length=255, blank=True, null=True)  # Tags sous forme de texte brut séparé par virgule (améliorable)

    def __str__(self):
        return self.title

