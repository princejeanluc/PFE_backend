import datetime
from core.models import New, Source, Crypto
from django.utils import timezone
from django.utils.dateparse import parse_datetime
from urllib.parse import urlparse
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
import re

def clean_text(text):
    # Supprime les emojis et caractères non BMP (Basic Multilingual Plane)
    emoji_pattern = re.compile(
        "[" 
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symboles et pictogrammes
        "\U0001F680-\U0001F6FF"  # transports et cartes
        "\U0001F1E0-\U0001F1FF"  # drapeaux
        "\U00002702-\U000027B0"  # symboles divers
        "\U000024C2-\U0001F251"  # autres symboles
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

def create_driver():
    options = Options()
    options.add_argument("--headless")   # headless moderne
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    
    # Important: fake un vrai navigateur
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/115.0.0.0 Safari/537.36"
    )
    
    # Empêche Selenium d'être détecté
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)

    driver = webdriver.Chrome(options=options)
    
    # Supprime la propriété webdriver
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    
    return driver



def store_articles(articles):
    i = 0
    for art in articles:
        # 1. Gérer la source
        source_name = art["source"].strip() if art.get("source") else "Unknown"
        source_url = f"https://{source_name}"
        source_obj, _ = Source.objects.get_or_create(
            name=source_name,
            defaults={"url": source_url}
        )

        # 2. Vérifier si l’article existe
        if New.objects.filter(url=art["url"]).exists():
            continue
        
        # 3. Convertir datetime
        try:
            pub_date = parse_datetime(art["datetime"])
            if pub_date is None:
                # Fallback si format non ISO
                pub_date = datetime.datetime.strptime(
                    art["datetime"].split(" GMT")[0], 
                    "%a %b %d %Y %H:%M:%S"
                )
            # Rendre "aware"
            if timezone.is_naive(pub_date):
                pub_date = timezone.make_aware(pub_date, timezone.get_current_timezone())
        except Exception:
            pub_date = timezone.now()

        # 4. Créer la news
        news = New.objects.create(
            title=art["title"],
            url=art["url"],
            datetime=pub_date,
            source=source_obj
        )
        

        # 5. Associer les cryptos
        for sym in art["cryptos"]:
            try:
                crypto_obj = Crypto.objects.get(symbol__iexact=sym)
                news.cryptos.add(crypto_obj)
            except Crypto.DoesNotExist:
                continue

        news.save()
        i+=1
    print(f"{i} articles enregistrés depuis cryptopanic.com")







def scroll_and_collect(driver, articles_seen, news_list):
    scroll_area = driver.find_element(By.CSS_SELECTOR, ".news-container")
    driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", scroll_area)
    time.sleep(1)
    
    articles = driver.find_elements(By.CSS_SELECTOR, ".news-row")
    for art in articles:
        try:
            title_el = art.find_element(By.CSS_SELECTOR, ".nc-title")
            title_text = clean_text(title_el.text.strip())
            a = art.find_element(By.CSS_SELECTOR, "a")
            link = a.get_attribute("href")
            if not title_text or "sponsored" in title_text.lower() or link in articles_seen:
                continue

            datetime_el = art.find_element(By.CSS_SELECTOR, ".nc-date time")
            datetime_str = datetime_el.get_attribute("datetime") if datetime_el else None

            source_el = art.find_element(By.CSS_SELECTOR, ".si-source-name")
            source = source_el.text if source_el else None

            cryptos = [
                c.text for c in art.find_elements(By.CSS_SELECTOR, ".nc-currency a.colored-link") if c.text
            ]

            news_list.append({
                "title": title_text,
                "url": link,
                "datetime": datetime_str,
                "source": source,
                "cryptos": cryptos
            })
            articles_seen.add(link)

        except Exception:
            continue
    return articles_seen , news_list



def fetch_news_from_cryptopanic():
    URL_NEWS_CRYPTOPANIC = "https://cryptopanic.com/news/"
    driver = create_driver()
    driver.get(URL_NEWS_CRYPTOPANIC)

    wait = WebDriverWait(driver, 10)
    articles_seen = set()

    news_list = []
    for _ in range(1):
        k , tmp_news  = scroll_and_collect(driver, articles_seen, news_list)
        news_list += tmp_news
    driver.quit()
    store_articles(news_list)
    print(f"{len(news_list)} articles traités pour enregistrement ... ")


