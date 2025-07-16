import os
import django
import sys
from dotenv import load_dotenv
load_dotenv()


def init_django():
    # 1. Ajouter POSA_backend au PYTHONPATH
    #BACKEND_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '...', 'POSA_backend'))
    #sys.path.append(BACKEND_PATH)
    # 2. Définir le module de configuration Django
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', os.getenv("DJANGO_SETTINGS_MODULE"))  # ← ici on utilise config.settings
    os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
    django.setup()