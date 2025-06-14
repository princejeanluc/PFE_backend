from django.db import models

class Crypto(models.Model):
    name = models.CharField(max_length=100)
    symbol = models.CharField(max_length=10)
    slug = models.SlugField(max_length=100, unique=True)
    img_url = models.URLField(blank=True, null=True)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return self.name


class CryptoInfo(models.Model):
    crypto = models.ForeignKey(Crypto, on_delete=models.CASCADE, related_name='market_data')  # Association: "have"
    price_usd = models.DecimalField(max_digits=20, decimal_places=8)
    volume_24h = models.DecimalField(max_digits=20, decimal_places=2)
    market_cap_usd = models.DecimalField(max_digits=20, decimal_places=2)
    percent_change_1h = models.FloatField()
    percent_change_24h = models.FloatField()
    percent_change_7d = models.FloatField()
    circulating_supply = models.DecimalField(max_digits=30, decimal_places=2)
    total_supply = models.DecimalField(max_digits=30, decimal_places=2)
    max_supply = models.DecimalField(max_digits=30, decimal_places=2, null=True, blank=True)
    is_active = models.BooleanField(default=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.crypto.symbol} @ {self.timestamp}"


class Source(models.Model):
    name = models.CharField(max_length=255)
    url = models.URLField()

    def __str__(self):
        return self.name


class News(models.Model):
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

