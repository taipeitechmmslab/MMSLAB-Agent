from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from state import TravelState

# 定義 routing function route，回傳下一個要執行的 node 名稱
def route(state: TravelState):  # 輸入 TravelState 讀取 critique 與 revisions 欄位
    # 取出 reflect node 的審核結果
    critique = state.get("critique")
    # 審核通過，或已經審核 3 輪，就結束
    if (critique is not None and critique["verdict"] == "pass") or state.get("revisions", 0) >= 3:
        return END
    # 沒通過就回到 planner node 重新規劃
    return "planner"

# 定義函式 build_graph，註冊四個 node 並組裝成可執行的 graph
def build_graph(retrieve_preferences, planner, executor, reflect):  # 輸入四個 node 函式
    # 以 TravelState 建立 graph
    builder = StateGraph(TravelState)

    # 加入四個 node
    builder.add_node("retrieve_preferences", retrieve_preferences)
    builder.add_node("planner", planner)
    builder.add_node("executor", executor)
    builder.add_node("reflect", reflect)

    # 依序連接檢索偏好、規劃、執行與審核
    builder.add_edge(START, "retrieve_preferences")
    builder.add_edge("retrieve_preferences", "planner")
    builder.add_edge("planner", "executor")
    builder.add_edge("executor", "reflect")
    # 在 reflect node 後方加入 conditional edge，由 route 決定回到 planner node 或結束
    builder.add_conditional_edges("reflect", route, ["planner", END])

    # 加上 MemorySaver 記住對話內容，再編譯 graph
    app = builder.compile(checkpointer=MemorySaver())
    # 顯示 graph 已建立完成
    print("StateGraph 建構完成")
    # 回傳 graph
    return app
