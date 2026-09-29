"""
Travel Agent - Tool 載入
=======================
tools.py 負責設定 Agent 可使用的 MCP tools，包含網路搜尋與天氣查詢。

執行流程：
    0. 載入套件
    1. 建立 tavily MCP server 設定，並讀取 TAVILY_API_KEY
    2. 建立 open-meteo MCP server 設定，供 Agent 查詢天氣資訊
    3. 使用 MultiServerMCPClient 啟動 MCP tool servers
    4. 取得 LangChain Agent 可使用的 tools 清單
    5. 回傳 MCP client 與 tools 給 main.py 使用

此模組提供 build_mcp_server_config() 與 load_mcp_tools() 函式供 main.py 呼叫。
"""

# 載入套件
import os
from langchain_mcp_adapters.client import MultiServerMCPClient


def build_mcp_server_config() -> dict:
    """回傳 MCP (Model Context Protocol) 工具設定，告訴程式要啟動哪些外部工具服務。

    MCP 是一個讓 LLM 與外部工具溝通的標準協定。
    每個 server 設定包含以下欄位：
        - command:   啟動該工具服務的執行檔（這裡用 npx 直接執行 npm 套件）
        - args:      傳給 command 的參數；-y 表示自動同意安裝、@latest 抓最新版
        - env:       要傳給該服務的環境變數（例如 API 金鑰）
        - transport: Agent 與 server 之間的溝通方式，stdio 代表透過標準輸入輸出
    """
    return {
        # tavily：提供網路搜尋功能，讓 Agent 能查詢即時資訊
        "tavily": {
            "command": "npx",
            "args": ["-y", "tavily-mcp@latest"],
            "env": {"TAVILY_API_KEY": os.getenv("TAVILY_API_KEY", "")},
            "transport": "stdio",
        },
        # open-meteo：提供免費天氣查詢服務，不需 API 金鑰
        "open-meteo": {
            "command": "npx",
            "args": ["-y", "open-meteo-mcp-server"],
            "transport": "stdio",
        },
    }


async def load_mcp_tools():
    """依照設定啟動 MCP 工具，並回傳 Agent 可以直接使用的工具清單。"""
    # 建立 MCP client，依照 server config 啟動外部工具服務
    client = MultiServerMCPClient(build_mcp_server_config())

    # 取得 LangChain Agent 可以直接使用的 tools 清單
    tools = await client.get_tools()

    # 工具錯誤容錯：handle_tool_error 預設 False，工具失敗會拋例外導致整個程式崩潰；
    # 設成字串後改把錯誤轉成訊息回饋 LLM，讓它換參數重試或略過，不中斷程式。
    for tool in tools:
        tool.handle_tool_error = "工具執行失敗，請改用其他參數重試，或在不依賴此工具的情況下繼續完成任務。"

    return client, tools
