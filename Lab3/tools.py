import os
from langchain_mcp_adapters.client import MultiServerMCPClient

# 定義函式 build_mcp_server_config，設定 Tavily、Open-Meteo 與 Frankfurter 三個 MCP server
def build_mcp_server_config() -> dict:
    return {
        # tavily：上網搜尋景點、住宿、交通資訊
        "tavily": {
            "command": "npx",  # 用 npx 啟動工具
            "args": ["-y", "tavily-mcp@0.2.22"],  # 指定工具版本
            # 傳入 Tavily 的 API Key
            "env": {"TAVILY_API_KEY": os.getenv("TAVILY_API_KEY", "")},
            "transport": "stdio",  # 在本機執行
        },
        # open-meteo：查詢天氣，不需要 API Key
        "open-meteo": {
            "command": "npx",  # 用 npx 啟動工具
            "args": ["-y", "open-meteo-mcp-server@2.0.1"],  # 指定工具版本
            "transport": "stdio",  # 在本機執行
        },
        # frankfurter：查詢匯率，直接連到網路上的服務
        "frankfurter": {
            "url": "https://mcp.frankfurter.dev/",  # 服務的網址
            "transport": "http",  # 透過網路連線
        },
    }

# 定義函式 _tool_error_hint，工具出錯時，告訴 LLM 該怎麼修正
def _tool_error_hint(error: Exception) -> str:  # 輸入工具的錯誤訊息
    # 回傳錯誤訊息和修正建議，讓 Agent 自己重試
    return (
        f"工具呼叫失敗：{error}\n"
        "請檢查並修正參數後重試（注意互斥參數不可同時帶入，例如 forecast_days "
        "與 start_date/end_date 只能擇一）。"
    )

# 定義函式 load_mcp_tools，以 asynchronous 方式在背景啟動 MCP 工具並取得工具清單
async def load_mcp_tools():
    # 依照上面的設定建立 MCP Client
    client = MultiServerMCPClient(build_mcp_server_config())

    # 以 await 取回背景 MCP 的工具清單
    tools = await client.get_tools()

    # 每個工具出錯時，都改成回傳錯誤訊息，不讓程式中斷
    for tool in tools:
        tool.handle_tool_error = _tool_error_hint

    # 顯示取得幾個工具
    print(f"已組裝 {len(tools)} 個 MCP 工具")
    # 回傳 MCP Client 與工具清單
    return client, tools
