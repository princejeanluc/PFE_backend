import os 

mcp_config = {
    "mcpServers": {
        "LLMAdapterTools": {
            "transport": "http",  # or "sse" 
            "url": os.getenv("MCP_SERVER_HOST"),
        }
    }
}