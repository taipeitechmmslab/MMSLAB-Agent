"""
CrewAI 個人化旅遊規劃 - Task 與 Crew 組裝
=================================
crew.py 負責建立每輪對話使用的通用 Task，並將一位 Manager Agent
與景點行程、住宿、天氣交通三位 Worker Agent 組成 Hierarchical Crew。

這個檔案不預先固定 Agent 的執行順序；使用者問題送入 Crew 後，
由 Manager Agent 判斷需要委派哪些 Worker Agent，再審查並整合結果。
"""

# ── 載入套件與 Agent 建立函數 ───────────────────────

from crewai import Crew, Process, Task
from crewai_tools.adapters.tool_collection import ToolCollection

from agents import build_agents, build_llm

# ── Task 行為規格 ────────────────────────────────────

# 通用 Task 接收本輪問題與對話背景，實際工作流程由 Manager 動態決定

USER_REQUEST_DESCRIPTION = (
    "今天是 {today}。使用者本輪的旅遊問題是：{query}\n\n"
    "先前對話背景如下：\n{conversation_context}\n\n"
    "本輪問題優先於先前內容；若本輪指涉先前內容，應依對話背景理解，"
    "最新修正覆蓋舊要求，也不要再次推薦已被否決的選項。"
    "依 Manager Agent 的動態分派與驗收規則完成本輪需求，最後直接回答原始問題。"
    "不要輸出內部委派、審查或重試過程。全文使用繁體中文與純文字；"
    "只有本輪使用者明確要求儲存、匯出或建立 Markdown 檔案時，"
    "準備交給寫檔工具的內容才可以使用 Markdown；第一次規劃不得自行匯出。"
    "一般最終回答不得超過一千五百字，中文字、數字、標點、空白與網址全部計入。"
)

# expected_output 定義 Crew 最終交付內容必須具備的品質與格式
USER_REQUEST_EXPECTED_OUTPUT = (
    "一份只回答本輪問題、同時正確承接先前對話的繁體中文答案。"
    "所有可能變動的資訊必須經過查證；最終答案依景點、住宿、天氣與交通分類保留能直接支持內容的"
    "實際來源網址，只寫工具或服務名稱不算來源。每類保留最直接的一至兩個網址，查不到就標示需確認，"
    "不得臆測。完整行程先說明從 Knowledge 歸納的偏好與安排理由，至少建立兩個偏好與具體安排的"
    "明確對應，再分段說明三天的每日安排及簡短理由，加上住宿區域、交通與天氣重點、預算及"
    "一項注意事項。"
    "終端回答使用純文字且不得超過一千五百字，中文字、數字、標點、空白與網址全部計入。"
    "Markdown 匯出成功時，只輸出 exports/ 開頭的 .md 相對路徑。"
)

# ── 組裝 Hierarchical Crew ──────────────────────────

def build_crew(tools: ToolCollection) -> Crew:
    """建立本輪要執行的 Hierarchical Crew。"""
    # 先建立共用 LLM，再建立一位 Manager Agent 與三位 Worker Agent
    manager, workers = build_agents(build_llm(), tools)

    # 建立 Crew，將通用 Task、Manager 與 Worker 組成層級式多 Agent 團隊
    return Crew(
        # agents 只放入可被委派的 Worker，Manager 由 manager_agent 指定
        agents=workers,
        # 每輪只有一張通用入口 Task，實際子工作由 Manager 現場判斷
        tasks=[
            Task(
                # description 會在 kickoff 時帶入日期、問題與先前對話
                description=USER_REQUEST_DESCRIPTION,
                # expected_output 定義最終答案必須符合的品質與格式
                expected_output=USER_REQUEST_EXPECTED_OUTPUT,
                # 不指定專責 Agent，由 Manager 依本輪問題動態選擇 Worker
            )
        ],
        # 使用 Hierarchical Process，讓 Manager 負責動態委派與整合
        process=Process.hierarchical,
        # 指定本輪負責管理三位 Worker 的 Manager Agent
        manager_agent=manager,
        # 顯示 Crew、Agent 與工具的完整執行面板
        verbose=True,
        # 關閉可能插入下一輪輸入提示的背景 Trace
        tracing=False,
    )
