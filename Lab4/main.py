from dotenv import load_dotenv

# 讀取 .env 裡的設定
load_dotenv()

import asyncio
from agent import build_agent
from chat import run_chat
from rag.router import build_router_query_engine
from tools import load_mcp_tools

# 定義函式 main，準備好所有元件後開始對話
async def main():
    # 啟動 MCP 工具：Tavily 搜尋與 Open-Meteo 天氣
    tools = await load_mcp_tools()

    # 建立 RouterQueryEngine
    router_engine = build_router_query_engine()

    # 建立旅遊 Agent
    agent = build_agent(tools)

    # 開始終端機對話
    await run_chat(router_engine, agent)

# 執行 main 函式，啟動整個程式
if __name__ == "__main__":
    asyncio.run(main())
