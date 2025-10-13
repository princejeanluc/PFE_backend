# mcp_server.py
import os, httpx
from datetime import datetime
from fastmcp import FastMCP
from typing import Literal

RelationMatrixType = Literal["spearman", "granger"]
RelationMatrixPeriod = Literal["7d", "14d", "30d"]
RelationMatrixLag = Literal[1,2,3,4,5]

API_BASE = os.getenv("API_BASE", "http://web:8000/api")
mcp = FastMCP(name="posamcp")

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

@mcp.tool(description="donne l'instant actuelle , utile pour comparer des éléments en prenant l'aspect emporelle")
def get_time_now()->str:
    return f"il est {datetime.now()}"

@mcp.tool(description="récupérer la liste des portefeuilles de l'utilisateur")
def list_portfolios():
    with httpx.Client(timeout=60, headers=_headers()) as cx:
        r = cx.get(f"{API_BASE}/llm/portfolios/list/")
        r.raise_for_status()
        return r.json()

@mcp.tool(description="Donne les informations sur un portefeuille à partir de son identifiant")
def portfolio_summary(portfolio_id: int):
    with httpx.Client(timeout=60, headers=_headers()) as cx:
        r = cx.get(f"{API_BASE}/llm/portfolio/{portfolio_id}/summary/")
        r.raise_for_status()
        return r.json()

@mcp.tool(description="retourne la liste des articles récents, leur date de publication et leur titre")
def recent_article_titles(limit: int = 50, since_hours: int = 168, lang: str = "fr") -> list[dict]:
    """
    Retourne uniquement (title, url, source, published_at) pour limiter les tokens.
    Wrappe: GET {API_BASE}/news/latest/?limit=&since_hours=&lang=
    """
    with httpx.Client(timeout=60, headers=_headers()) as cx:
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

@mcp.tool(name="get_market_metrics", description="Retourne les métriques récentes sur le comportement du marché (volatilité, concentration, volume, etc.)")
def get_market_metrics() -> list[dict]:
    with httpx.Client(timeout=60, headers=_headers()) as cx:
        r = cx.get(f"{API_BASE}/llm/market-metrics/")
        r.raise_for_status()
        return r.json()
    results = []
    for ind in r:
        results.append({
                    "indicator": ind.name,
                    "indicatorValue": ind.value,
                    "message": ind.message,
                    "colorFlag": ind.flag,
                })
    return results
    
@mcp.tool(name="get_crypto_map", 
          description="Retourne la cartographie du marché de la crypto sur 30 jours en utilisant une réduction de dimension et du clustering, donne des informations marchés et associe les comportements commun par cluster")
def get_crypto_map() -> dict:
    with httpx.Client(timeout=60, headers=_headers()) as cx:
        r = cx.get(f"{API_BASE}/crypto-map/")
        r.raise_for_status()
        return r.json()


@mcp.tool(name="get_latest_info", 
          description="les dernires informations du marché et les prévisions horaires suivantes par les modèles xgboost et GRU. (NB : le modèle GRU est plus stable car il n'explose pas au niveau des pertes)")
def get_latest_info() -> dict:
    with httpx.Client(timeout=60, headers=_headers()) as cx:
        r = cx.get(f"{API_BASE}/cryptos/latest-info")
        r.raise_for_status()
        return r.json()


@mcp.tool(name="get_relation_map", description="Retourne soit la matrice de causalité , soit la matrice de corrélation ('spearman'|'ranger') sur une période (en jour ex: '30d' -> 30 jours) choisi et avec un lag (en heure ex: 1 -> 1h) défini (uniquement pour causalité)")
def get_relation_map(type:RelationMatrixType , period:RelationMatrixPeriod , lag:RelationMatrixLag) -> dict:
    with httpx.Client(timeout=60, headers=_headers()) as cx:
        r = cx.get(f"{API_BASE}/crypto-relations/", 
                   params={
                       type  :type, 
                       period: period, 
                       lag: lag
                   })
        r.raise_for_status()
        return r.json()

@mcp.tool
def _debug_auth():
    return {"has_token": bool(_AUTH["tool_token"]), "model_key": bool(_AUTH["model_key"])}

if __name__ == "__main__":
    mcp.run(transport="http",
    host="0.0.0.0", 
    port=1445,
    log_level="DEBUG",
    )
