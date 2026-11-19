from typing import Literal
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from state import TravelState

# 定義 Critique 類別，規定 LLM 回傳審核結果的格式
class Critique(BaseModel):  # Critique 類別繼承 BaseModel 父類別
    # verdict 只能是 pass（通過）或 revise（要修改）
    verdict: Literal["pass", "revise"] = Field(description="通過或需修正")
    # issues 列出要修改的問題，通過時是空的
    issues: list[str] = Field(default_factory=list, description="各面向發現的問題")

# 定義函式 build_reflect_prompt，建立 reflect node 的審核 Prompt
def build_reflect_prompt() -> str:
    return """\
你是旅遊行程結果審核員，請判斷目前行程是否已經足夠回答使用者需求。

請檢查：
1. 需求符合度：是否回應目的地、日期、天數、預算與其他明確條件。
2. 偏好一致性：是否符合偏好檔案中的景點、住宿、預算與交通習慣。
3. 完整性：是否包含每日行程、交通、住宿建議、預算與必要提醒。
4. 合理性：時間安排、地理動線、費用加總與前後內容是否有明顯矛盾。
5. 格式：是否符合每日行程與預算摘要的輸出要求。

若結果完整且沒有必須修改的問題，回傳 verdict="pass"，issues 留空。
只有當問題會影響主要行程，而且能靠修改目前草案解決時，才回傳 verdict="revise"。
issues 最多列 2 項，每項用一句話清楚指出要修改的內容。

不要自行查證或斷言即時票價、天氣、營業時間等外部資訊。
使用者未提供的出發機場、航班，以及即時房價、空房或取消條件，不可作為 revise 理由。
草案若已誠實標示費用為估算或待確認，也不可只因缺乏即時資料判 revise。
若只有提醒事項而沒有必要修改，必須判 pass。"""

# 定義函式 create_reflect，建立 reflect node 並回傳
def create_reflect(llm):  # 輸入 LLM
    # 讓 LLM 依照 Critique 的格式回傳
    critic = llm.with_structured_output(Critique, method="function_calling")

    # 定義 node reflect，檢查最新的行程草案
    async def reflect(state: TravelState) -> dict:  # 以 asynchronous 方式執行，輸入 TravelState 讀取 messages、preferences 與 revisions 欄位
        # 先設成空字串
        draft = ""
        # 從最新的訊息往回找 executor node 寫的行程草案
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                draft = msg.content
                # 找到就停止
                break

        # 從 messages 找出使用者最新的一句話，找不到時就用空字串
        user_query = next(
            (msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)),
            "",
        )

        # 把審核規則、使用者需求、偏好與草案交給背景的 LLM 審核，前景以 await 取回審核結果
        critique = await critic.ainvoke([
            SystemMessage(content=build_reflect_prompt()),  # 審核規則
            HumanMessage(content=(
                f"[使用者需求]\n{user_query}\n\n"  # 使用者這輪的需求
                f"[使用者偏好檔案]\n{state.get('preferences') or '無'}\n\n"  # 使用者的偏好
                f"[待審核的行程草案]\n{draft}"  # 要審核的行程草案
            )),
        ])

        # 把審核結果轉成 dict，寫入 state 的 critique
        # 已審核的輪次加一
        return {"critique": critique.model_dump(), "revisions": state.get("revisions", 0) + 1}

    # 回傳這個 node，交給 main.py 使用
    return reflect
