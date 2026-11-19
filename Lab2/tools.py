import os
from langchain_mcp_adapters.client import MultiServerMCPClient

# 定義函式 build_mcp_server_config，設定 Tavily 與 Open-Meteo 兩個 MCP server
def build_mcp_server_config() -> dict:
    return {
        # tavily：上網搜尋景點、住宿、交通資訊
        "tavily": {
            "command": "npx",  # 用 npx 啟動工具
            "args": ["-y", "tavily-mcp@latest"],  # 使用最新版的工具
            # 傳入 Tavily 的 API Key
            "env": {"TAVILY_API_KEY": os.getenv("TAVILY_API_KEY", "")},
            "transport": "stdio",  # 在本機執行
        },
        # open-meteo：查詢天氣，不需要 API Key
        "open-meteo": {
            "command": "npx",  # 用 npx 啟動工具
            "args": ["-y", "open-meteo-mcp-server"],  # 指定要啟動的工具
            "transport": "stdio",  # 在本機執行
        },
    }

# 定義函式 load_mcp_tools，啟動 MCP 工具並取得工具清單
async def load_mcp_tools():
    # 依照上面的設定建立 MCP Client
    client = MultiServerMCPClient(build_mcp_server_config())

    # 等 MCP 工具啟動完成，取得工具清單
    tools = await client.get_tools()

    # 每個工具出錯時，都改成回傳錯誤訊息給 LLM，不讓程式中斷
    for tool in tools:
        tool.handle_tool_error = "工具執行失敗，請改用其他參數重試，或在不依賴此工具的情況下繼續完成任務。"

    # 回傳 MCP Client 與工具清單
    return client, tools
