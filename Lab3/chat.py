from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from openai import APIError
from prompt import read_query

# 定義函式 preview_text，文字太長時只顯示前面一段
def preview_text(value, max_chars: int = 400) -> str:  # 輸入要顯示的內容與最多顯示幾個字
    # 先轉成文字
    text = str(value).strip()
    # 沒超過就全部顯示
    if len(text) <= max_chars:
        return text
    # 超過就只顯示前面，並標出總字數
    return f"{text[:max_chars].rstrip()}...（共 {len(text)} 字元）"

# 定義函式 print_header，印出標題
def print_header(title: str):  # 輸入標題文字
    # 上下各印一條分隔線
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")

# 定義函式 _print_update，顯示每個 node 的執行結果
def _print_update(namespace, data):  # 輸入更新的來源與內容
    # 來自 executor node 內部的更新：顯示工具的呼叫與結果
    if namespace:
        for value in data.values():
            if not isinstance(value, dict):
                continue
            for message in value.get("messages", []):
                if isinstance(message, AIMessage) and message.tool_calls:
                    # 顯示呼叫了哪個工具
                    for tool_call in message.tool_calls:
                        print(f"\n呼叫工具: {tool_call['name']}")
                        print(f"   參數：{preview_text(tool_call['args'], 120)}")
                # 顯示工具回傳的結果
                elif isinstance(message, ToolMessage):
                    print(f"工具回傳: {preview_text(message.content, 150)}")
        return

    # 依照 node 的名稱，顯示各自的結果
    for node_name, output in data.items():
        # retrieve_preferences node：顯示找到的偏好
        if node_name == "retrieve_preferences":
            preferences = output.get("preferences")
            if preferences:
                print_header("Retrieve Preferences — 偏好前置檢索")
                print('輸出欄位：state["preferences"]（偏好原文片段）')
                print(preview_text(preferences, 600))
        # planner node：顯示計畫
        elif node_name == "planner":
            print_header("Planner — 規劃 / 修訂計畫")
            print('輸出欄位：state["plan"]（可修改的計畫物件）')
            for i, step in enumerate(output["plan"], 1):
                print(f"  {i}. {step}")
            print_header("Executor — 執行與生成")
        # executor node：顯示行程草案
        elif node_name == "executor":
            for message in reversed(output["messages"]):
                if isinstance(message, AIMessage) and message.content:
                    print(message.content)
                    break
        # reflect node：顯示審核結果
        elif node_name == "reflect":
            critique = output["critique"]
            print_header("Reflect — 多面向品質檢查")
            print('輸出欄位：state["critique"]（structured output）')
            print(f"  verdict：{critique['verdict']}")
            print(f"  審核輪次：{output.get('revisions', 0)}")
            if critique["issues"]:
                print("  issues：")
                for issue in critique["issues"]:
                    print(f"    ・{issue}")

# 定義函式 run_chat，開始多輪對話，直到使用者結束
async def run_chat(graph):  # 輸入 graph
    # 固定 thread_id，讓每一輪都接在同一段對話裡
    config = {"configurable": {"thread_id": "travel-session-1"}}

    # 顯示歡迎訊息與範例問題
    print(
        """
==================================================
個人化旅遊規劃 Agentic AI（LangGraph）已就緒
輸入旅遊需求開始規劃，輸入 'exit' 或 'quit' 結束
範例：
   1. 幫我安排下周二三天兩夜的大阪的古蹟參訪行程
   2. 幫我把 Day 2 改成以室內景點為主
==================================================
"""
    )

    # 從第 1 輪開始
    turn = 1
    # 不斷接收使用者輸入，直到使用者結束
    while True:
        # 讀取使用者輸入
        user_input = read_query(turn)
        # 使用者要結束對話
        if user_input is None:
            print("\n再見")
            break
        # 只按 Enter 就重新等待輸入
        if not user_input:
            continue

        # 準備這輪要交給 graph 的資料
        # plan、critique、revisions 每輪都重新開始
        payload = {
            "messages": [HumanMessage(content=user_input)],  # 使用者這輪的問題
            "plan": [],  # 清除上一輪的計畫
            "critique": None,  # 清除上一輪的審核結果
            "revisions": 0,  # 已審核的輪次歸零
        }

        # API 出錯時只中止這一輪，不讓程式關閉
        try:
            # 執行 graph，並即時取得每個 node 的結果
            async for namespace, data in graph.astream(
                payload, config, stream_mode="updates", subgraphs=True
            ):
                # 顯示收到的結果
                _print_update(namespace, data)
        # API 出錯時顯示錯誤訊息
        except APIError as error:
            print(f"\n\nNVIDIA API 呼叫失敗，這輪規劃未完成：{error}")
            print("   通常是端點限流或服務暫時過載，稍等一兩分鐘再重問一次即可。\n")
        else:
            # 這輪規劃完成
            print("\n\n本輪規劃完成\n")

        # 輪次加一
        turn += 1
