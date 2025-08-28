import asyncio
from dotenv import load_dotenv
load_dotenv()
from fastmcp import Client as MCPClient
import os
from google import genai
from google.genai import types as gtypes

SYSTEM = """
Tu es un analyste crypto rigoureux. Stratégie :
1) Utilise recent_articles puis read_recent_articles(limit: int = 20, since_hours: int = 48, lang: str = "fr") pour 15–25 articles (24–48h).
2) Résume l'actualité.
3) Fais une synthèse marché + 3 conseils prudentiels.
Règles: français, factuel, cite source (média + URL), pas de promesses.
"""



async def run_advice(since_hours=48, limit=20, lang="fr", risk_profile="prudent"):
    mcp = MCPClient("mcp_server.py")
    gem = genai.Client(api_key=os.getenv("GOOGLE_AI_API_KEY"))
    prompt = f"Analyse les actualités crypto (since_hours={since_hours}, limit={limit}, lang={lang}, profil={risk_profile})."
    async with mcp:
        response = await gem.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=gtypes.GenerateContentConfig(
                system_instruction=SYSTEM,
                temperature=0.2,
                tools=[mcp.session],
                max_output_tokens=3072,
            )
        )
        print(response.text)
        
    return response




if __name__ == "__main__":
    asyncio.run(run_advice(
            since_hours=48,
            limit=20,
            lang="fr",
            risk_profile="prudent",
        ))
    
