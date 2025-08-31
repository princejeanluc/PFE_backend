# cryptopanic.py
import os
import re
import time
import datetime

# --- Initialiser Django AVANT d'importer les modèles ---
#os.environ.setdefault("DJANGO_SETTINGS_MODULE", os.getenv("DJANGO_SETTINGS_MODULE", "config.settings"))
#import django  # noqa: E402
#django.setup()  # noqa: E402

from django.utils import timezone  # noqa: E402
from django.utils.dateparse import parse_datetime  # noqa: E402
from core.models import New, Source, Crypto  # noqa: E402

from selenium import webdriver  # noqa: E402
from selenium.webdriver.chrome.options import Options  # noqa: E402
from selenium.webdriver.common.by import By  # noqa: E402
from selenium.webdriver.support.ui import WebDriverWait  # noqa: E402
from selenium.webdriver.support import expected_conditions as EC  # noqa: E402
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException  # noqa: E402


def clean_text(text: str) -> str:
    emoji_pattern = re.compile(
        "[" 
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictograms
        "\U0001F680-\U0001F6FF"  # transport
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"  # misc
        "\U000024C2-\U0001F251"  # others
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub("", text or "").strip()


def create_driver() -> webdriver.Chrome:
    opts = Options()
    # Headless moderne + réglages container
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    )
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)

    driver = webdriver.Chrome(options=opts)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return driver


def _parse_datetime(s: str) -> datetime.datetime:
    """
    Supporte ISO-8601 (avec Z) et fallback texte Cryptopanic.
    Retourne un datetime *aware* en timezone serveur.
    """
    if not s:
        return timezone.now()
    try:
        dt = parse_datetime(s)  # gère 2025-08-30T12:34:56Z
        if dt is None:
            # Exemple: "Sat Aug 30 2025 12:34:56 GMT"
            dt = datetime.datetime.strptime(s.split(" GMT")[0], "%a %b %d %Y %H:%M:%S")
        if timezone.is_naive(dt):
            dt = timezone.make_aware(dt, timezone.get_current_timezone())
        return dt
    except Exception:
        return timezone.now()


def store_articles(articles: list[dict]) -> int:
    saved = 0
    for art in articles:
        url = art.get("url")
        title = clean_text(art.get("title", ""))
        if not url or not title:
            continue

        # Source
        source_name = (art.get("source") or "Unknown").strip()
        if not source_name.startswith("http"):
            source_url = f"https://{source_name}"
        else:
            source_url = source_name
        source_obj, _ = Source.objects.get_or_create(name=source_name, defaults={"url": source_url})

        # Unicité par URL
        if New.objects.filter(url=url).exists():
            continue

        news = New.objects.create(
            title=title,
            url=url,
            datetime=_parse_datetime(art.get("datetime")),
            source=source_obj,
        )

        for sym in art.get("cryptos", []):
            try:
                c = Crypto.objects.get(symbol__iexact=sym)
                news.cryptos.add(c)
            except Crypto.DoesNotExist:
                pass

        saved += 1
    return saved


def collect_once(driver, seen: set[str]) -> list[dict]:
    """Collecte une page/scroll et renvoie UNIQUEMENT les nouveaux items."""
    out: list[dict] = []

    # S’assurer que la liste est présente
    WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, ".news-container"))
    )

    # Scroll pour charger plus d’items
    container = driver.find_element(By.CSS_SELECTOR, ".news-container")
    driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", container)
    time.sleep(1)

    for row in driver.find_elements(By.CSS_SELECTOR, ".news-row"):
        try:
            a = row.find_element(By.CSS_SELECTOR, "a")
            link = a.get_attribute("href")
            if not link or link in seen:
                continue

            title_el = row.find_element(By.CSS_SELECTOR, ".nc-title")
            title = clean_text(title_el.text)
            if not title or "sponsored" in title.lower():
                continue

            dt_el = row.find_element(By.CSS_SELECTOR, ".nc-date time")
            dt_str = dt_el.get_attribute("datetime") if dt_el else None

            src_el = row.find_element(By.CSS_SELECTOR, ".si-source-domain")
            source = (src_el.text or "Cryptopanic") if src_el else "Cryptopanic"

            cryptos = [c.text for c in row.find_elements(By.CSS_SELECTOR, ".nc-currency a.colored-link") if c.text]

            out.append({"title": title, "url": link, "datetime": dt_str, "source": source, "cryptos": cryptos})
            seen.add(link)
        except (NoSuchElementException, StaleElementReferenceException):
            continue
        except Exception:
            continue
    return out


def fetch_news_from_cryptopanic(scrolls: int = 3) -> None:
    URL = "https://cryptopanic.com/news/"
    driver = create_driver()
    try:
        driver.get(URL)
        # Première peinture
        WebDriverWait(driver, 15).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".news-row"))
        )

        seen: set[str] = set()
        collected: list[dict] = []
        for _ in range(scrolls):
            batch = collect_once(driver, seen)
            if not batch:
                break
            collected.extend(batch)

    finally:
        driver.quit()

    saved = store_articles(collected)
    print(f"{len(collected)} articles trouvés, {saved} enregistrés depuis cryptopanic.com.")


if __name__ == "__main__":
    fetch_news_from_cryptopanic()
