# mcp_server.py
import os, httpx
from fastmcp import FastMCP

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000/api")
mcp = FastMCP(name="LLMAdapterTools")

# Mémoire process-locale (propre à CE serveur MCP)
_AUTH = {"tool_token": "", "model_key": ""}

@mcp.tool
def _auth_set(tool_token: str, model_key: str | None = None) -> dict:
    """
    Initialise l'auth pour CETTE session MCP.
    À appeler depuis l'orchestrateur AVANT de donner la main au LLM.
    """
    _AUTH["tool_token"] = tool_token or ""
    if model_key:
        _AUTH["model_key"] = model_key
    return {"ok": True}

def _headers() -> dict:
    return {
        "X-Model-Key": _AUTH["model_key"] or os.getenv("LLM_MODEL_KEY", "CHANGE_ME"),
        "X-Tool-Token": _AUTH["tool_token"] or "",
    }

@mcp.tool
def list_portfolios():
    with httpx.Client(timeout=15, headers=_headers()) as cx:
        r = cx.get(f"{API_BASE}/llm/portfolios/list/")
        r.raise_for_status()
        return r.json()

@mcp.tool
def portfolio_summary(portfolio_id: int):
    with httpx.Client(timeout=20, headers=_headers()) as cx:
        r = cx.get(f"{API_BASE}/llm/portfolio/{portfolio_id}/summary/")
        r.raise_for_status()
        return r.json()

@mcp.tool
def recent_article_titles(limit: int = 50, since_hours: int = 168, lang: str = "fr") -> list[dict]:
    """
    Retourne uniquement (title, url, source, published_at) pour limiter les tokens.
    Wrappe: GET {API_BASE}/news/latest/?limit=&since_hours=&lang=
    """
    with httpx.Client(timeout=15, headers=_headers()) as cx:
        r = cx.get(f"{API_BASE}/news/latest/", params={
            "limit": limit, "since_hours": since_hours, "lang": lang
        })
        r.raise_for_status()
        items = r.json()
    out = []
    for it in items:
        out.append({
            "title": it.get("title"),
            "url": it.get("url"),
            "source": it.get("source") or it.get("publisher") or it.get("feed"),
            "published_at": it.get("datetime") or it.get("published_at"),
        })
    return out


# (debug optionnel)
@mcp.tool
def _debug_auth():
    return {"has_token": bool(_AUTH["tool_token"]), "model_key": bool(_AUTH["model_key"])}

if __name__ == "__main__":
    mcp.run()
