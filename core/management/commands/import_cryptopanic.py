import csv
from django.utils.dateparse import parse_datetime
from django.core.management.base import BaseCommand
from yourapp.models import New, Source, Crypto

class Command(BaseCommand):
    help = "Importe les actualités depuis un fichier CSV CryptoPanic"

    def add_arguments(self, parser):
        parser.add_argument('csv_file', type=str, help='Chemin vers le fichier CSV')
        parser.add_argument('--crypto-symbol', type=str, help='Symbole de la crypto associée (ex: BTC)', required=True)

    def handle(self, *args, **options):
        file_path = options['csv_file']
        symbol = options['crypto_symbol'].upper()

        try:
            crypto = Crypto.objects.get(symbol=symbol)
        except Crypto.DoesNotExist:
            self.stderr.write(f"[ERREUR] Crypto avec le symbole '{symbol}' non trouvée.")
            return

        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            imported = 0

            for row in reader:
                # Chercher ou créer la source
                source, _ = Source.objects.get_or_create(
                    name=row['sourceId'],
                    defaults={'url': row['sourceUrl']}
                )

                # Parser la date
                try:
                    news_datetime = parse_datetime(row['newsDatetime'])
                except Exception:
                    self.stderr.write(f"[AVERTISSEMENT] Date invalide pour l'entrée : {row['newsDatetime']}")
                    continue

                # Vérifier si la news existe déjà
                if New.objects.filter(url=row['url']).exists():
                    continue

                # Créer l'article de news
                New.objects.create(
                    crypto=crypto,
                    source=source,
                    title=row['title'],
                    url=row['url'],
                    datetime=news_datetime,
                    summary=row['description'] or "",
                    positive=int(row.get('positive') or 0),
                    negative=int(row.get('negative') or 0),
                    tags=None  # ou extraire depuis `row` si pertinent
                )
                imported += 1

        self.stdout.write(f"[SUCCÈS] {imported} articles importés pour {crypto.name}.")
