# helpers_llm.py
from datetime import datetime, timedelta
import jwt, os
def mint_tool_token(user_id: int, scope=("portfolio:read",), ttl_s=90) -> str:
    claims = {
        "sub": str(user_id),
        "aud": "llm",
        "scope": list(scope),
        "exp": datetime.utcnow() + timedelta(seconds=ttl_s),
    }
    return jwt.encode(claims, os.getenv("SECRET_KEY"), algorithm="HS256")