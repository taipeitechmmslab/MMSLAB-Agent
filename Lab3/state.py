from langgraph.graph import MessagesState

# 定義 state 類別 TravelState，存放所有 node 都會用到的資料
class TravelState(MessagesState):  # TravelState 類別繼承 MessagesState 父類別
    preferences: str          # retrieve_preferences node 檢索到的偏好資料
    plan: list[str]           # planner node 排好的步驟
    critique: dict | None     # reflect node 的審核意見
    revisions: int            # reflect node 已審核的輪次
