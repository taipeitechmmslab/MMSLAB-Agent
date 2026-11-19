from crewai import Crew, Process, Task
from crewai_tools.adapters.tool_collection import ToolCollection
from agents import build_agents, build_llm

# 通用 Task 的任務描述，帶入今天日期、本輪問題與先前對話背景
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

# 通用 Task 的期望產出，定義最終答案的品質與格式
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

# 定義函式 build_crew，建立本輪要執行的 Crew
def build_crew(tools: ToolCollection) -> Crew:  # 輸入 MCP 工具清單
    # 建立共用 LLM，再建立 Manager Agent 與三個專責 Agent
    manager, workers = build_agents(build_llm(), tools)

    # 把 Task、Manager Agent 與三個專責 Agent 組成 Crew
    return Crew(
        agents=workers,  # 只放入可以被分派的三個專責 Agent
        # 每輪只有一張通用 Task，實際要分派給誰由 Manager Agent 判斷
        tasks=[
            Task(
                description=USER_REQUEST_DESCRIPTION,  # 任務描述
                expected_output=USER_REQUEST_EXPECTED_OUTPUT,  # 期望產出
            )
        ],
        process=Process.hierarchical,  # 使用層級模式，由 Manager Agent 分派與整合
        manager_agent=manager,  # 指定 Manager Agent
        verbose=True,  # 顯示 Crew、Agent 與工具的執行過程
        tracing=False,  # 關閉背景追蹤，避免訊息插入對話畫面
    )
