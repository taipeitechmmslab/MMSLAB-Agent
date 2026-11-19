from datetime import date
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from state import TravelState

# 定義 Plan 類別，規定 LLM 要回傳一份步驟清單
class Plan(BaseModel):  # Plan 類別繼承 BaseModel 父類別
    # 每個元素是一個步驟
    steps: list[str] = Field(description="有序的執行計畫，每個元素是一個可執行步驟")

# 定義函式 build_plan_prompt，建立 planner node 的規劃 Prompt
def build_plan_prompt() -> str:
    return """\
你是旅遊規劃的任務分析師，負責把使用者的旅遊需求拆解成一份有序的執行計畫。
計畫每一步是一個可執行動作，後續會交給具備工具（搜尋/天氣/匯率）的 executor 依序執行。
標準步驟骨架（可依需求增減、調整順序）：
1. 查目的地天氣以決定室內外安排
2. 查匯率以換算預算與成本
3. 依偏好檔案搜尋景點、住宿、交通的即時資訊
4. 估算總成本並做預算可行性判斷
5. 生成每日行程草案

計畫必須反映偏好檔案：把「優先安排的類型」與「必須避開的類型」直接寫進搜尋與生成步驟
（例如偏好在地文創、不愛過度觀光化商圈，步驟就寫「搜尋在地文創景點與巷弄美食，避開純觀光商圈」），
不要寫「熱門景點」「必去景點」這類與偏好無關的通用步驟。

若收到上一輪的 critique，請針對 issues 逐項調整計畫（例如預算超支就加入「改用平價住宿/景點方案」步驟，動線不順就加入「重排地理動線」步驟），不要原封不動重列。

輸出規則：
- steps 的每個元素就是一個步驟的完整描述，不要加編號前綴
- 只產出計畫，不要執行步驟內容、不要解釋"""

# 定義函式 create_planner，建立 planner node 並回傳
def create_planner(llm):  # 輸入 LLM
    # 讓 LLM 依照 Plan 的格式回傳
    plan_llm = llm.with_structured_output(Plan, method="function_calling")

    # 定義 node planner，根據偏好與使用者需求排出執行步驟
    async def planner(state: TravelState) -> dict:  # 以 asynchronous 方式執行，輸入 TravelState 讀取 messages、preferences 與 critique 欄位
        # 取出上一輪的審核結果，第一次規劃時沒有
        critique = state.get("critique")
        # 取出審核發現的問題
        issues = (critique or {}).get("issues")
        # 有問題代表這輪是修改，沒有問題代表第一次規劃
        is_revision = bool(issues)
        # 把問題逐條列出
        critique_block = "\n".join(f"・{i}" for i in issues) if is_revision else "無（這是第一次規劃）"

        # 修改時只改有問題的地方；第一次規劃則排出完整計畫
        task = (
            "請只針對上方審核問題，產出「修正步驟」清單：每步對應一個 issue 寫「改什麼」。"
            "天氣、匯率、以及上一版已完成的景點/住宿/交通蒐集都不要重列——"
            "executor 會保留上一版草案，只改你列出的地方。"
            if is_revision else
            "請針對上方對話中最新一則旅遊需求，產出本輪執行計畫。"
        )

        # 把今天日期、偏好、審核問題與任務整理成這輪的指令
        directive = HumanMessage(content=(
            f"[今天日期] {date.today().isoformat()}（規劃涉及「下週二」等相對日期時以此為準）\n\n"
            f"[使用者偏好檔案]\n{state.get('preferences') or '無'}\n\n"
            f"[上一輪審核發現的問題]\n{critique_block}\n\n"
            + task
        ))

        # 把規則、對話歷史和指令交給背景的 LLM 產生計畫，前景以 await 取回計畫
        plan = await plan_llm.ainvoke([
            SystemMessage(content=build_plan_prompt()),  # 規劃規則
            *state["messages"],  # 對話歷史
            directive,  # 這輪的指令
        ])
        # 把步驟寫入 state 的 plan，交給 executor node
        return {"plan": plan.steps}

    # 回傳這個 node，交給 main.py 使用
    return planner
