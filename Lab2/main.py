from dotenv import load_dotenv

# 讀取 .env 裡的設定
load_dotenv()

import asyncio
from datetime import datetime
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from agent import build_agent
from chat import run_chat
from rag import build_retriever
from template import build_preference_query
from tools import load_mcp_tools

# 定義函式 format_docs，把檢索到的 Document 整理成一段文字
def format_docs(docs):  # 輸入 RAG 檢索器找到的 Document 列表
    # 每筆前面加上「- 」，並用空行隔開
    return "\n\n".join(f"- {doc.page_content}" for doc in docs)

# 定義函式 build_agent_input，把今天日期、過往偏好與使用者問題組成 Agent 的輸入訊息
def build_agent_input(data: dict) -> dict:  # 輸入 context 與 question
    # 取得今天日期，讓 Agent 算得出「下週」是哪幾天
    today = datetime.now().strftime("%Y-%m-%d")
    # 依照 Agent 需要的 messages 格式回傳
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

# 定義函式 build_rag_agent_chain，用 LCEL 把 RAG 檢索與 Agent 串成一條 chain
def build_rag_agent_chain(retriever, agent):  # 輸入 RAG 檢索器與 Agent
    return (
        # 同時準備 context 與 question 兩份資料
        {
            # context：把問題改寫成檢索字串，找出偏好資料，再整理成文字
            "context": RunnableLambda(build_preference_query) | retriever | RunnableLambda(format_docs),
            # question：使用者的問題原封不動傳下去
            "question": RunnablePassthrough(),
        }
        # 把 context 與 question 組成 Agent 的輸入訊息
        | RunnableLambda(build_agent_input)
        # 交給 Agent 產生回答
        | agent
    )

# 定義函式 main，準備好所有元件後開始對話
async def main():
    # 啟動 MCP 工具：Tavily 搜尋與 Open-Meteo 天氣
    mcp_client, tools = await load_mcp_tools()

    # 保留 MCP Client，讓對話期間可以一直使用工具
    _ = mcp_client

    # 建立 RAG 檢索器
    retriever = build_retriever()

    # 建立旅遊 Agent
    agent = build_agent(tools)

    # 把 RAG 檢索與 Agent 串成一條 chain
    rag_agent_chain = build_rag_agent_chain(retriever, agent)

    # 開始終端機對話
    await run_chat(rag_agent_chain)

# 執行 main 函式，啟動整個程式
if __name__ == "__main__":
    asyncio.run(main())
