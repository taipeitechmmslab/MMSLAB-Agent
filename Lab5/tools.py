import os
from crewai_tools import MCPServerAdapter
from crewai_tools.adapters.tool_collection import ToolCollection
from mcp import StdioServerParameters

# 定義函式 start_mcp_tools，啟動四個 MCP Server 並取得工具清單
def start_mcp_tools() -> tuple[MCPServerAdapter, ToolCollection]:
    # 用 MCPServerAdapter 一次管理兩個本機 MCP Server 與兩個遠端 MCP Server
    adapter = MCPServerAdapter(
        [
            # Tavily：上網搜尋景點與旅遊資訊
            StdioServerParameters(
                command="npx",  # 用 npx 啟動工具
                args=["-y", "tavily-mcp@0.2.21"],  # 固定版本，避免套件更新後行為改變
                # 傳入 Tavily 的 API Key
                env={"TAVILY_API_KEY": os.getenv("TAVILY_API_KEY", "")},
            ),
            # MoodTrip Hotel：查詢指定日期的飯店空房與房價，不需要 API Key
            {
                "url": "https://api.moodtrip.ai/api/mcp-http",  # 遠端 MCP Server 的位址
                "transport": "streamable-http",  # 透過 HTTP 連線
            },
            # Weather：查詢天氣預報
            StdioServerParameters(
                command="npx",  # 用 npx 啟動工具
                # 固定版本，避免工具清單改變
                args=["-y", "@dangahagan/weather-mcp@1.13.0"],
            ),
            # Frankfurter：查詢匯率
            {
                "url": "https://mcp.frankfurter.dev/",  # 官方提供的遠端 MCP Server
                "transport": "streamable-http",  # 透過 HTTP 連線
            },
        ],
        # 只載入本 Lab 需要的五個工具
        "tavily_search",  # Tavily 的搜尋工具
        "search_hotels_with_rates",  # 飯店房價搜尋工具
        "get_hotel_details",  # 飯店詳細資料工具
        "get_forecast",  # 天氣預報工具
        "get_rates",  # 匯率查詢工具
        # 第一次用 npx 下載可能超過 30 秒，所以延長等待時間
        connect_timeout=120,
    )
    # 取得工具清單，讓 Agent 可以用工具名稱取出工具
    tools = adapter.tools

    # 顯示載入的工具數量與名稱，確認 MCP 是否成功啟動
    print(f"已載入 {len(tools)} 個 MCP 工具：{[t.name for t in tools]}")

    # 回傳 adapter 與工具清單，adapter 交給 main.py 負責關閉
    return adapter, tools
