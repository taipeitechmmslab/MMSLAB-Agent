"""
Travel Agent - 終端機輸出
=======================
chat.py 負責顯示啟動畫面、串流各節點執行進度，並驅動多輪對話迴圈。

程式流程：
  1. 顯示啟動畫面與旅遊問題範例。
  2. 從終端機接收使用者輸入。
  3. 使用 graph.astream() 接收頂層節點與 Executor 子圖更新。
  4. 顯示偏好、計畫、工具活動、行程草案與審核結果。
  5. NVIDIA API 呼叫失敗時只中止該輪，讓使用者稍後重新提問。
  6. 保留同一個 thread_id，持續處理多輪對話直到使用者結束。

串流使用 stream_mode="updates" + subgraphs=True：
    - 頂層更新：節點寫入 State 的 preferences、plan、messages 與 critique。
    - 子圖更新：Executor 內部 Agent 產生的工具呼叫與回傳。

"""

# ── 載入套件 ──────────────────────────────────────────────
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from openai import APIError

from prompt import read_query


# ── 終端機文字顯示輔助函式 ────────────────────────────────
def preview_text(value, max_chars: int = 400) -> str:
    """截斷過長文字，避免偏好內容或工具參數洗版終端機。"""
    # 將輸入轉成字串，方便統一計算與截斷長度。
    text = str(value).strip()
    # 文字未超過上限時直接完整回傳。
    if len(text) <= max_chars:
        return text
    # 超過上限時只保留開頭，並標示原始總字數。
    return f"{text[:max_chars].rstrip()}...（共 {len(text)} 字元）"


def print_header(title: str):
    """印出節點區段標題，讓使用者知道 graph 跑到哪個階段。"""
    # 前後各印一條 60 字元分隔線，讓不同節點的輸出容易辨識。
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


# ── 顯示 Graph 與 Executor 子圖更新 ───────────────────────
def _print_update(namespace, data):
    """依更新來源顯示工具呼叫或各節點的輸出。"""
    # namespace 有內容代表更新來自 Executor 子圖。
    if namespace:
        # 子圖更新中的 AIMessage 可能包含一個或多個工具呼叫。
        for value in data.values():
            if not isinstance(value, dict):
                continue
            for message in value.get("messages", []):
                if isinstance(message, AIMessage) and message.tool_calls:
                    # 只顯示工具名稱與簡短參數，不顯示冗長的原始回傳。
                    for tool_call in message.tool_calls:
                        print(f"\n🔧 呼叫工具: {tool_call['name']}")
                        print(f"   參數：{preview_text(tool_call['args'], 120)}")
                # ToolMessage 代表工具已執行完成，只顯示截斷後的回傳內容。
                elif isinstance(message, ToolMessage):
                    print(f"✅ 工具回傳: {preview_text(message.content, 150)}")
        return

    # 頂層更新的 key 是節點名稱，value 是該節點寫入 State 的欄位。
    for node_name, output in data.items():
        # Retrieve Preferences：顯示檢索到的偏好片段。
        if node_name == "retrieve_preferences":
            preferences = output.get("preferences")
            if preferences:
                print_header("Retrieve Preferences — 偏好前置檢索")
                print('📚 輸出欄位：state["preferences"]（偏好原文片段）')
                print(preview_text(preferences, 600))
        # Planner：顯示計畫，並先印出下一個 Executor 節點標題。
        elif node_name == "planner":
            print_header("Planner — 規劃 / 修訂計畫")
            print('輸出欄位：state["plan"]（可修改的計畫物件）')
            for i, step in enumerate(output["plan"], 1):
                print(f"  {i}. {step}")
            print_header("Executor — 執行與生成")
        # Executor：只顯示最後一則完整 AI 回答。
        elif node_name == "executor":
            for message in reversed(output["messages"]):
                if isinstance(message, AIMessage) and message.content:
                    print(message.content)
                    break
        # Reflect：顯示結構化審核結果與審核輪次。
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


# ── 啟動多輪終端機對話 ────────────────────────────────────
async def run_chat(graph):
    """啟動多輪對話介面，直到使用者主動結束。"""
    # 固定 thread_id 讓 MemorySaver 把每輪輸入接在同一段旅遊對話中。
    config = {"configurable": {"thread_id": "travel-session-1"}}

    # 顯示程式啟動訊息、結束方式與兩個多輪對話範例。
    print(
        """
==================================================
🧳 個人化旅遊規劃 Agentic AI（LangGraph）已就緒
💡 輸入旅遊需求開始規劃，輸入 'exit' 或 'quit' 結束
💡 範例：
   1. 幫我安排下周二三天兩夜的大阪的古蹟參訪行程
   2. 幫我把 Day 2 改成以室內景點為主
==================================================
"""
    )

    # 第一輪從 1 開始顯示，之後每完成一次規劃就加一。
    turn = 1
    # 持續接收輸入，直到使用者輸入結束關鍵字或按下 Ctrl+C / Ctrl+D。
    while True:
        # 呼叫 prompt.py 顯示目前輪次並讀取一行輸入。
        user_input = read_query(turn)
        # None 代表使用者要求結束對話。
        if user_input is None:
            print("\n👋 再見")
            break
        # 空字串代表使用者只按 Enter，不執行 Graph 並繼續等待輸入。
        if not user_input:
            continue

        # 建立本輪 Graph 輸入；messages 使用 MessagesState 的 reducer 自動附加到歷史。
        # plan、critique 與 revisions 屬於單輪暫存資料，因此每輪開始時重置。
        payload = {
            "messages": [HumanMessage(content=user_input)],  # 本輪新的使用者需求
            "plan": [],  # 清除上一輪執行計畫
            "critique": None,  # 清除上一輪審核結果
            "revisions": 0,  # 審核輪次重新從 0 開始
        }

        # 將整輪 Graph 執行包在 try/except，避免 NVIDIA API 暫時過載時終止整個程式。
        try:
            # 訂閱節點更新，並開啟 subgraphs 取得 Executor 內部工具呼叫。
            async for namespace, data in graph.astream(
                payload, config, stream_mode="updates", subgraphs=True
            ):
                # 每收到一個更新就依頂層或子圖來源顯示對應內容。
                _print_update(namespace, data)
        # 只捕捉模型端點回傳的 API 錯誤；其他程式錯誤仍保留 traceback，方便除錯。
        except APIError as error:
            print(f"\n\n⚠️ NVIDIA API 呼叫失敗，這輪規劃未完成：{error}")
            print("   通常是端點限流或服務暫時過載，稍等一兩分鐘再重問一次即可。\n")
        else:
            # astream 正常結束代表本輪已通過審核或達到最大審核輪次。
            print("\n\n✅ 本輪規劃完成\n")

        # 下一次輸入顯示新的輪次編號。
        turn += 1
