# core/utils/email_check.py
import re, os
import dns.resolver

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Option: ajoute des domaines via env: DISPOSABLE_EMAIL_DOMAINS="mailinator.com,yopmail.com"
DISPOSABLE = {
    "mailinator.com","10minutemail.com","tempmail.com","guerrillamail.com","yopmail.com",
    *[d.strip().lower() for d in os.getenv("DISPOSABLE_EMAIL_DOMAINS","").split(",") if d.strip()]
}

def normalize_email(email: str) -> str:
    return (email or "").strip().lower()

def is_valid_syntax(email: str) -> bool:
    return bool(EMAIL_RE.match(email or ""))

def is_disposable(email: str) -> bool:
    dom = (email.split("@")[-1] if "@" in email else "").lower()
    return dom in DISPOSABLE

def mx_exists(domain: str) -> bool:
    try:
        dns.resolver.resolve(domain, "MX")
        return True
    except Exception:
        # certains domaines n’ont pas de MX mais ont un A
        try:
            dns.resolver.resolve(domain, "A")
            return True
        except Exception:
            return False
