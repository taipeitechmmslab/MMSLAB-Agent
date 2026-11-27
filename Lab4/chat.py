from datetime import datetime
from llama_index.core.agent.workflow import ToolCall, ToolCallResult
from llama_index.core.workflow import Context
from openai import APIError
from prompt import read_query

# 定義函式 run_query，先用 RouterQueryEngine 檢索偏好，再交給 Agent 回答
async def run_query(router_engine, agent, ctx, query: str):  # 輸入檢索器、Agent 與問題
    # 提示使用者 Agent 正在處理
    print("\nAgent 思考中...\n")
    try:
        # 由 RouterQueryEngine 選擇 Index 並檢索偏好
        print("RouterQueryEngine 檢索中...")
        rag_response = router_engine.query(query)

        # 取出選路結果
        selector_result = (rag_response.metadata or {}).get("selector_result")
        if selector_result:
            # 工具編號對應的 Index 名稱
            tool_names = {
                1: "SummaryIndex（整體偏好）",
                2: "VectorStoreIndex（特定細節）",
                3: "DocumentSummaryIndex（整趟紀錄回顧）",
                4: "KeywordTableIndex（精確名稱命中）",
            }
            # 顯示選了哪一個 Index 與選擇的理由，編號從 1 開始
            for sel in selector_result.selections:
                choice = sel.index + 1
                print(f"Router 選路結果：choice ({choice}) {tool_names.get(choice, '')}")
                print(f"理由：{sel.reason}")

        # 顯示檢索到的偏好摘要，只顯示前 200 個字
        rag_text = str(rag_response)
        print(f"偏好摘要：{rag_text[:200]}...")
        print()

        # 把今天日期、偏好摘要與使用者問題組成 Agent 的輸入
        today = datetime.now().strftime("%Y-%m-%d")
        agent_input = (
            f"今天日期：{today}\n\n"
            f"我過往的台灣旅遊紀錄顯示我的偏好：\n{rag_text}\n\n"
            f"使用者問題：{query}"
        )

        # 執行 Agent，並帶入前幾輪的對話記憶
        handler = agent.run(agent_input, ctx=ctx)

        # 邊執行邊接收事件，顯示呼叫的工具與工具回傳的結果
        async for event in handler.stream_events():
            if isinstance(event, ToolCall):
                print(f"呼叫工具: {event.tool_name}({event.tool_kwargs})")
            elif isinstance(event, ToolCallResult):
                print(f"工具回傳: {str(event.tool_output)[:150]}...")

        # 等 Agent 完成，顯示最後的回答
        response = await handler
        print(f"\n{response}")
    except APIError as error:
        # API 呼叫失敗時只放棄這一輪，不讓程式中斷
        print(f"\nNVIDIA API 呼叫失敗，這輪回答未完成：{error}")
        print("通常是端點限流（429/503），稍等一兩分鐘再重問一次即可。")

# 定義函式 run_chat，啟動多輪對話，直到使用者結束
async def run_chat(router_engine, agent):  # 輸入 RouterQueryEngine 與 Agent
    # 建立 Context，讓 Agent 記得同一段對話
    ctx = Context(agent)

    # 顯示啟動畫面與範例問題
    print("""
==================================================
旅遊規劃助理已就緒（支援國內/國外規劃）
輸入問題開始對話，輸入 'exit' 或 'quit' 結束
範例：
1. [KeywordTableIndex] 我之前住過的「木門厝」有哪些住宿特色？請記住這種風格，之後用來挑大阪住宿
2. [DocumentSummaryIndex] 那我那趟去台南的行程詳細的景點安排和心得是什麼
3. [VectorStoreIndex] 接著我想走沿溪谷、溪水清澈的健行步道。請參考我過去走過的類似地點，找出大阪附近符合這類特色的健行路線
4. [SummaryIndex] 最後請從我過去的旅遊紀錄中，歸納我的旅遊偏好並統計我過去旅行平均一天安排幾個景點、整體步調多快，作為大阪行程的安排節奏
==================================================
""")

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
        # 執行這一輪的查詢
        await run_query(router_engine, agent, ctx, query)
        print()
        # 進入下一輪
        turn += 1
