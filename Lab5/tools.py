"""
CrewAI 個人化旅遊規劃－MCP 工具連線
=================================

tools.py 負責建立 MCP（Model Context Protocol）連線，
讓 Agent 能透過統一協定呼叫外部的景點搜尋、住宿、天氣預報與匯率查詢服務。

本實驗使用兩種 MCP Server：

1. 本機 Stdio Server
   由 npx 啟動，透過標準輸入與輸出交換資料。
   - command：啟動服務使用的執行檔
   - args：傳給執行檔的參數與固定套件版本
   - env：傳入服務需要的環境變數，例如 API Key

2. 遠端 HTTP Server
   直接連線至服務提供的網址。
   - url：MCP Server 的連線位址
   - transport：資料傳輸方式，本實驗使用 streamable-http

本實驗連接的 MCP Server：
- Tavily MCP：景點與旅遊資訊搜尋
- MoodTrip Hotel MCP：住宿與房價查詢
- Weather MCP：天氣預報查詢
- Frankfurter MCP：匯率查詢
"""

# ── 載入套件 ────────────────────────────────────────

import os

from crewai_tools import MCPServerAdapter
from crewai_tools.adapters.tool_collection import ToolCollection
from mcp import StdioServerParameters


# ── 啟動 MCP Server 並載入旅遊資訊工具 ──────────────────────

def start_mcp_tools() -> tuple[MCPServerAdapter, ToolCollection]:
    """啟動四個 MCP Server，回傳生命週期 Adapter 與原始工具清單。"""
    # 建立 MCPServerAdapter，統一管理兩個本機 Stdio Server 與兩個遠端 HTTP Server
    adapter = MCPServerAdapter(
        [
            # Tavily MCP：供景點行程、住宿及天氣交通 Agent 查詢即時資訊
            StdioServerParameters(
                # 固定 Tavily MCP 版本，避免套件更新後行為或 schema 改變
                command="npx",
                args=["-y", "tavily-mcp@0.2.21"],
                # 將 .env 中的 Tavily API Key 傳入 MCP 子程序
                env={"TAVILY_API_KEY": os.getenv("TAVILY_API_KEY", "")},
            ),
            # MoodTrip Hotel MCP：查詢指定日期的飯店空房、即時房價與住宿資料
            {
                # 公開搜尋不需要 API Key，使用者完成訂房仍須前往外部網站
                "url": "https://api.moodtrip.ai/api/mcp-http",
                # CrewAI 直接使用遠端 Server 的 streamable-http 傳輸方式
                "transport": "streamable-http",
            },
            # Weather MCP：查詢全球天氣預報
            StdioServerParameters(
                # 固定 Weather MCP 版本，避免工具清單與 schema 自動改變
                command="npx",
                args=["-y", "@dangahagan/weather-mcp@1.13.0"],
            ),
            # Frankfurter MCP：透過遠端 Server 查詢匯率
            {
                # 使用官方託管的 MCP Server，不需要啟動本機子程序
                "url": "https://mcp.frankfurter.dev/",
                # 指定遠端 Server 使用 streamable-http 傳輸方式
                "transport": "streamable-http",
            },
        ],
        # 只載入本 Lab 需要的 Tavily 搜尋工具
        "tavily_search",
        # 只載入 Hotel MCP 的飯店即時房價搜尋與候選詳情工具
        "search_hotels_with_rates",
        "get_hotel_details",
        # 只載入 Weather MCP 的天氣預報工具
        "get_forecast",
        # 只載入 Frankfurter MCP 的匯率查詢工具
        "get_rates",
        # 第一次透過 npx 下載固定版本可能超過預設 30 秒，因此延長連線等待時間
        connect_timeout=120,
    )
    # 保留 Adapter 回傳的 ToolCollection，讓 Agent 能直接用工具名稱取得工具
    tools = adapter.tools

    # 顯示實際載入的工具數量與名稱，方便讀者確認 MCP 是否成功啟動
    print(f"已載入 {len(tools)} 個 MCP 工具：{[t.name for t in tools]}")

    # Adapter 交給 main.py 管理啟動與關閉，工具清單則提供給 Crew 與 Agent 使用
    return adapter, tools
