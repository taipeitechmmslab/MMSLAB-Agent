from dotenv import load_dotenv
# 讀取 .env 裡的設定
load_dotenv()
import asyncio
import os
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI
from chat import run_chat
from executor import create_executor
from graph import build_graph
from planner import create_planner
from rag import build_retriever
from reflect import create_reflect
from retrieve_preferences import create_retrieve_preferences
from tools import load_mcp_tools

# 定義函式 main，準備好所有元件後開始對話
async def main():
    # 啟動 MCP 工具：搜尋、天氣與匯率
    mcp_client, tools = await load_mcp_tools()

    # 建立搜尋旅遊紀錄的檢索器
    retriever = build_retriever()

    # 建立 LLM，使用 NVIDIA NIM 的模型
    llm = ChatOpenAI(
        # NVIDIA NIM 的網址
        base_url="https://integrate.api.nvidia.com/v1",
        # 從 .env 讀取 API Key
        api_key=os.getenv("NVIDIA_API_KEY"),
        # 從 .env 讀取模型名稱
        model=os.getenv("CHAT_MODEL"),
        # 最多等待 5 分鐘
        stream_chunk_timeout=300,
        # 失敗時最多重試 2 次
        max_retries=2,
        # 限制呼叫速度，避免超過 API 的使用上限
        rate_limiter=InMemoryRateLimiter(requests_per_second=0.15, max_bucket_size=1),
    )

    # 建立四個 node
    retrieve_preferences = create_retrieve_preferences(retriever)
    planner = create_planner(llm)
    executor = create_executor(llm, tools)
    reflect = create_reflect(llm)

    # 組裝 graph
    graph = build_graph(retrieve_preferences, planner, executor, reflect)

    # 開始對話
    await run_chat(graph)

if __name__ == "__main__":
    # 執行 main
    asyncio.run(main())
