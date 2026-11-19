from langchain_core.messages import HumanMessage
from state import TravelState
from template import build_preference_query

# 定義函式 create_retrieve_preferences，建立 retrieve_preferences node 並回傳
def create_retrieve_preferences(retriever):  # 輸入 rag.py 建立的檢索器
    # 定義 node retrieve_preferences，找出使用者的旅遊偏好
    async def retrieve_preferences(state: TravelState) -> dict:  # 以 asynchronous 方式執行，輸入 TravelState 讀取 messages 欄位
        # 從對話歷史找出使用者最新的一句話
        # 找不到時就用空字串
        user_query = next(
            (msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)),
            "",
        )

        # 用使用者的問題到 Milvus 搜尋相關的旅遊紀錄
        docs = retriever.invoke(build_preference_query(user_query))
        # 把找到的段落接起來，寫入 state 的 preferences
        return {"preferences": "\n\n".join(doc.page_content for doc in docs)}

    # 回傳這個 node，交給 main.py 使用
    return retrieve_preferences
