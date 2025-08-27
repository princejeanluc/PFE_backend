from fastmcp import FastMCP
import shared.db as db
db.init_django()
from core.models import New

from core.serializers import NewSerializer

mcp_server = FastMCP(name="MCP POSA Server")

@mcp_server.tool
def read_recent_articles(limit: int = 20, since_hours: int = 48, lang: str = "fr") -> list[dict]:
    """
    Retourne des articles récents .
    """
    news = New.objects.order_by('-datetime')[:limit]
    serialized = NewSerializer(news, many=True)
    return serialized.data





if __name__ == "__main__":
    mcp_server.run()