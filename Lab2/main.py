"""
Travel Agent - 主程式入口
=======================
main.py 負責把偏好檢索、Agent、外部工具與終端機介面串成完整旅遊規劃流程。

執行流程：
    0. 載入套件與環境變數
    1. 載入 Agent 可以使用的 MCP 外部工具
    2. 建立可以搜尋過往旅遊紀錄的 retriever
    3. 建立負責回答問題的旅遊 Agent
    4. 使用 LCEL 串接偏好搜尋、資料整理與 Agent 回答
    5. 啟動終端機互動介面，接收使用者多輪問題

執行方式：
    python main.py
"""

# 載入套件與環境變數工具
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

# 載入套件
import asyncio
from datetime import datetime

from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from agent import build_agent
from chat import run_chat
from rag import build_retriever
from template import build_preference_query
from tools import load_mcp_tools


def format_docs(docs):
    """將 Milvus 檢索回來的 Documents 整理成字串"""
    return "\n\n".join(f"- {doc.page_content}" for doc in docs)


def build_agent_input(data: dict) -> dict:
    """將今天日期、過往偏好與使用者問題組成 Agent 的輸入訊息。"""
    today = datetime.now().strftime("%Y-%m-%d")
    return {
        "messages": [
            {
                "role": "user",
                "content": (
                    f"今天日期：{today}\n\n"
                    f"我過往的台灣旅遊紀錄顯示我的偏好：\n{data['context']}\n\n"
                    f"使用者問題：{data['question']}"
                ),
            }
        ]
    }


# LCEL 流程：使用者問題 -> 偏好搜尋文字 -> Milvus 檢索 -> Agent 輸入 -> Agent 回答
def build_rag_agent_chain(retriever, agent):
    """使用 LCEL 串接偏好查詢、檢索結果整理與 Agent 回答。"""
    return (
        # 平行處理兩條線：context 走 RAG 檢索、question 原樣保留，供下一步合併
        {
            # context 線：build_preference_query 把問題改寫成偏好搜尋詞
            #            -> retriever 從 Milvus 找回相關 Documents
            #            -> format_docs 整理成 Agent 可閱讀的純文字
            "context": RunnableLambda(build_preference_query) | retriever | RunnableLambda(format_docs),
            # question 線：原樣傳遞使用者問題，留給 build_agent_input 組訊息
            "question": RunnablePassthrough(),
        }
        # 將今天日期、context（過往偏好）與 question（本輪問題）合併成 Agent 的輸入訊息
        | RunnableLambda(build_agent_input)
        # 交給旅遊 Agent 產生最終回答（含工具呼叫與短期記憶）
        | agent
    )


async def main():
    # 載入外部工具， Tavily 搜尋最新景點資訊和 open-meteo 查詢最新天氣資訊
    mcp_client, tools = await load_mcp_tools()

    # 保留工具連線，讓終端機對話期間可以持續呼叫工具
    _ = mcp_client

    # 建立RAG檢索器，用來從過往旅遊紀錄找出相關偏好
    retriever = build_retriever()

    # 建立旅遊 Agent，負責整合工具、偏好資料並產生回答
    agent = build_agent(tools)

    # 把偏好檢索流程和 Agent 回答流程串在一起
    rag_agent_chain = build_rag_agent_chain(retriever, agent)

    # 啟動終端機介面，讓使用者可以一輪一輪輸入問題
    await run_chat(rag_agent_chain)


if __name__ == "__main__":
    asyncio.run(main())
