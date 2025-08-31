import os 
mcp_config = {
    "mcpServers": {
        "posamcp": {
            "transport": "http",  # or "sse" 
            "url": os.getenv("MCP_SERVER_HOST"),
        }
    }
}