from prompt import read_query

# 定義函式 run_query，執行一輪問答，並在終端機即時顯示過程
async def run_query(rag_agent_chain, query: str, config: dict):  # 輸入 chain、問題與設定
    # 提示使用者 Agent 正在處理
    print("\nAgent 思考中...\n")

    # 用 astream_events 邊執行邊接收每個步驟的事件
    async for event in rag_agent_chain.astream_events(query, config=config, version="v2"):
        # 取出事件種類與內容
        event_type = event["event"]
        event_data = event["data"]

        # RAG 檢索完成：顯示從 Milvus 找到的偏好資料
        if event_type == "on_retriever_end":
            docs = event_data.get("output", [])
            print(f"從 Milvus 檢索到 {len(docs)} 筆偏好資料：")
            # 每筆只顯示前 80 個字，並把換行換成空白
            for i, doc in enumerate(docs, 1):
                snippet = doc.page_content[:80].replace("\n", " ")
                print(f"{i}. {snippet}...")
            print()

        # LLM 產生文字：一個字一個字顯示回答
        elif event_type == "on_chat_model_stream":
            content = event_data["chunk"].content
            # 有文字時才顯示，而且不換行
            if content:
                print(content, end="", flush=True)

        # Tool Calling 開始：顯示呼叫的工具與參數
        elif event_type == "on_tool_start":
            tool_name = event["name"]
            tool_input = event_data.get("input", "")
            print(f"\n呼叫工具: {tool_name}({tool_input})")

        # 工具回傳結果：只顯示前 150 個字
        elif event_type == "on_tool_end":
            print(f"工具回傳: {str(event_data.get('output', ''))[:150]}...")

    print()

# 定義函式 run_chat，啟動多輪對話，直到使用者結束
async def run_chat(rag_agent_chain):  # 輸入 chain
    # 固定 thread_id，讓 Agent 記得同一段對話
    config = {"configurable": {"thread_id": "trip-1"}}

    # 顯示啟動畫面與範例問題
    print(
        """
==================================================
旅遊規劃助理已就緒（支援國內/國外規劃）
輸入問題開始對話，輸入 'exit' 或 'quit' 結束
範例：
1. 幫我安排下周二三天兩夜的大阪的古蹟參訪行程
2. 京都的氛圍跟我去過的哪個台灣地方比較像？
==================================================
"""
    )

    # 從第 1 輪開始
    turn = 1
    # 一直對話，直到使用者結束
    while True:
        # 讀取這一輪的問題
        query = read_query(turn)

        # 使用者要結束對話，就說再見並離開迴圈
        if query is None:
            print("\n再見")
            break
        # 使用者只按 Enter，就跳過這一輪
        if not query:
            continue

        print()
        # 把問題交給 chain，並顯示回答
        await run_query(rag_agent_chain, query, config)
        print()
        # 進入下一輪
        turn += 1
