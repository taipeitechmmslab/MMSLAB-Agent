"""
Travel Agent - 終端機輸出
=======================
chat.py 負責顯示啟動畫面、串流模型回答、檢索結果與工具呼叫，並驅動多輪對話迴圈。

執行流程：
    0. 顯示啟動 banner 與範例問題
    1. 透過 prompt.read_query 接收使用者輸入
    2. 呼叫 rag_agent_chain.astream_events 串流接收 chain 事件
    3. 依事件類型分別顯示檢索結果、工具呼叫、模型回答
    4. 使用者結束輸入時印出告別訊息

此模組提供 run_chat() 函式供 main.py 呼叫。
"""

from prompt import read_query


async def run_query(rag_agent_chain, query: str, config: dict):
    """執行一次使用者問題，並在終端機顯示檢索結果、工具呼叫與回答內容。"""
    print("\n⏳ Agent 思考中...\n")

    # astream_events 會把 chain 內部每個步驟（retriever、LLM、tool）拆成事件即時吐出，
    # 這裡用 async for 邊跑邊收，依事件類型顯示對應資訊，使用者不必等全部結束才看到輸出
    async for event in rag_agent_chain.astream_events(query, config=config, version="v2"):
        event_type = event["event"]  # 事件種類，例如 on_retriever_end、on_chat_model_stream
        event_data = event["data"]   # 該事件的輸入/輸出內容

        if event_type == "on_retriever_end":
            # RAG 檢索完成事件：Milvus 向量資料庫已回傳與本輪問題最相關的偏好段落
            # 這裡把檢索結果摘要印出來，方便確認 RAG 是否抓到合適的過往紀錄
            docs = event_data.get("output", [])
            print(f"📚 從 Milvus 檢索到 {len(docs)} 筆偏好資料：")
            for i, doc in enumerate(docs, 1):
                # 只取前 80 字並把換行換成空白，避免終端機顯示破版
                snippet = doc.page_content[:80].replace("\n", " ")
                print(f"   {i}. {snippet}...")
            print()

        elif event_type == "on_chat_model_stream":
            # LLM 串流輸出事件：每產生一個 token 就觸發一次，讓使用者可以看到即時回覆
            content = event_data["chunk"].content
            if content:
                print(content, end="", flush=True)

        elif event_type == "on_tool_start":
            # 工具呼叫開始事件：Agent 決定要使用外部工具 Tavily 搜尋最新景點資訊和 open-meteo 查詢最新天氣資訊
            # 印出工具名稱與輸入參數，讓使用者知道 Agent 正在查什麼
            tool_name = event["name"]
            tool_input = event_data.get("input", "")
            print(f"\n🔧 呼叫工具: {tool_name}({tool_input})")

        elif event_type == "on_tool_end":
            # 工具回傳結果事件：外部工具已回應，內容會再丟回 LLM 做下一步推理
            # 工具回傳通常很長，這裡只截前 150 字供確認，避免洗版
            print(f"✅ 工具回傳: {str(event_data.get('output', ''))[:150]}...")

    print()


async def run_chat(rag_agent_chain):
    """啟動多輪對話介面，直到使用者主動結束。"""
    # 設定固定 thread_id，讓 Agent 可以保留多輪對話記憶
    config = {"configurable": {"thread_id": "trip-1"}}

    print(
        """
==================================================
🧳 旅遊規劃助理已就緒（支援國內/國外規劃）
💡 輸入問題開始對話，輸入 'exit' 或 'quit' 結束
💡 範例：
   1. 幫我安排下周二三天兩夜的大阪的古蹟參訪行程
   2. 京都的氛圍跟我去過的哪個台灣地方比較像？
==================================================
"""
    )

    # 持續接收使用者輸入，直到 read_query 回傳 None
    turn = 1
    while True:
        query = read_query(turn)

        if query is None:
            print("\n👋 再見")
            break
        if not query:
            continue

        print()
        # 將本輪問題交給 chain，串流顯示回答
        await run_query(rag_agent_chain, query, config)
        print()
        turn += 1
