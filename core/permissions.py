# permissions.py
import os, jwt
from rest_framework.permissions import BasePermission
from django.contrib.auth import get_user_model
from django.conf import settings
User = get_user_model()


class IsLLMRequest(BasePermission):
    """
    Exige:
      - X-Model-Key: clé d'application LLM (LLM_MODEL_KEY dans env/settings)
      - X-Tool-Token: JWT court (aud='llm') contenant:
          sub = user_id
          scope = liste de scopes (ex: ["portfolio:read","news:read"])
    Vérifie les scopes requis (overridable via `required_llm_scopes` sur la vue)
    et attache `request.user` au user issu du token.
    """
    message = "LLM key or tool-token invalid."
    default_required_scopes = {"portfolio:read", "news:read"}

    def has_permission(self, request, view):
        print(request.headers)
        # 1) Vérif clé d'app
        expected_key = os.getenv("LLM_MODEL_KEY") or getattr(settings, "LLM_MODEL_KEY", None)
        model_key = request.headers.get("X-Model-Key")
        if not expected_key or model_key != expected_key:
            self.message = "Invalid or missing X-Model-Key."
            return False

        # 2) Récup token outils
        tool_token = request.headers.get("X-Tool-Token")
        if not tool_token:
            self.message = "Missing X-Tool-Token."
            return False

        # 3) Scopes requis (surcharge possible au niveau de la vue)
        required_scopes = getattr(view, "required_llm_scopes", self.default_required_scopes)
        # 4) Décodage & vérifs
        try:
            claims = jwt.decode(
                tool_token,
                key=getattr(settings, "SECRET_KEY"),
                algorithms=["HS256"],
                audience="llm",
                options={"require": ["exp", "aud", "sub"]},
            )
        except Exception as e:
            self.message = f"Invalid tool token: {e}"
            print(self.message)
            return False

        token_scopes = set(claims.get("scope", []))
        if not set(required_scopes).issubset(token_scopes):
            self.message = "Insufficient LLM tool-token scope."
            return False

        # 5) Attacher l'utilisateur
        try:
            request.user = User.objects.get(pk=int(claims["sub"]))
        except Exception:
            self.message = "Unknown user in token."
            return False

        return True