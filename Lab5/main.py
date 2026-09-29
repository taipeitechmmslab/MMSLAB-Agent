"""
CrewAI 個人化旅遊規劃 - 主程式入口
=================================
main.py 負責啟動 MCP Server、接收使用者輸入，並透過 Conversational Flow
保存近期對話，再將每輪旅遊問題交給 Hierarchical Crew 處理。

執行方式：
  python main.py
"""

# ── 載入套件與專案模組 ──────────────────────────────

from datetime import date

from dotenv import load_dotenv

from crewai import Flow
from crewai.events.utils.console_formatter import ConsoleFormatter
from crewai.experimental.conversational import ConversationState
from crewai.flow import listen
from crewai_tools.adapters.tool_collection import ToolCollection

from crew import build_crew
from tools import start_mcp_tools

# 讀取 .env，載入模型、MCP 與 Knowledge 連線需要的環境變數
load_dotenv()


# ── 建立保存近期對話的 Conversational Flow ─────────────────

class TravelFlow(Flow[ConversationState]):
    """保存對話歷史，並將每輪問題交給 Hierarchical Crew。"""

    # 開啟 CrewAI 官方多輪對話功能，讓 Flow 自動保存 user 與 assistant 訊息
    conversational = True

    # 保存 main() 啟動的 MCP 工具清單，供每輪新建立的 Crew 共用
    mcp_tools: ToolCollection

    def route_turn(self, context: dict) -> str:
        """所有旅遊問題都交給 Crew，工作流程由 Manager 動態判斷。"""
        # 回傳 travel 事件名稱，觸發下方以 @listen("travel") 標記的處理函式
        return "travel"

    # 收到 route_turn() 回傳的 travel 事件後，執行本輪旅遊問答
    @listen("travel")
    def handle_travel(self) -> str:
        """建立本輪 Crew，帶入問題與最近兩輪對話後回傳最終答案。"""
        # Flow 已先加入本輪 user 訊息，因此排除最後一則，只取前兩輪作為對話背景
        history = self.conversation_messages[:-1][-4:]
        # 將訊息 list 整理成 Manager 能直接閱讀的純文字對話內容
        # 沒有先前對話時，則以「沒有先前對話」作為第一輪問題的背景
        conversation_context = "\n\n".join(
            # 每則訊息保留 role 與 content，讓 Agent 能分辨使用者和助理內容
            f"{message['role']}：{message['content']}" for message in history
        ) or "（沒有先前對話。）"

        # 每輪建立新的 Crew，避免上一輪的 Manager 與 Task 狀態影響後續委派
        crew = build_crew(self.mcp_tools)

        # 清除上一輪的完成事件，讓本輪重新等待 Crew 完成
        ConsoleFormatter.crew_completion_printed.clear()

        # 將本輪問題、今天日期與 Flow 保存的對話背景交給通用 Task
        result = crew.kickoff(
            inputs={
                "query": self.state.current_user_message,
                "today": date.today().isoformat(),
                "conversation_context": conversation_context,
            }
        )

        # 最多等待 10 秒，避免背景完成事件插入下一輪的輸入提示
        ConsoleFormatter.crew_completion_printed.wait(timeout=10)
        # 將 CrewOutput 轉成字串，交給 Flow 保存並顯示給使用者
        return str(result)


# ── 啟動個人化旅遊規劃系統 ──────────────────────────────

def main() -> None:
    """啟動 MCP 工具，並使用 Flow.chat() 進行多輪對話。"""
    # 啟動四個 MCP Server；程式執行期間只啟動一次，後續每輪 Crew 共用連線
    adapter, tools = start_mcp_tools()

    try:
        # 顯示系統名稱與三個可直接測試的示範問題
        print(
            """
==================================================
個人化旅遊規劃 Agentic AI（CrewAI）已就緒
範例問題：
   1. 幫我安排下週二三天兩夜的大阪古蹟參訪行程
   2. 幫我把行程調整成以室內景點為主
   3. 請把剛剛的行程匯出成 Markdown
==================================================
"""
        )

        # 建立 TravelFlow，將前面載入的 MCP 工具交給每輪 Crew 使用
        # chat() 負責輸入迴圈、對話歷史、提示文字與離開指令
        TravelFlow(
            # 保存 MCP 工具清單
            mcp_tools=tools,
            # 隱藏 Flow 自身事件，終端只保留對話提示與最終回答
            suppress_flow_events=True,
            # 關閉背景 Trace，避免額外輸出插入對話畫面
            tracing=False,
        ).chat(
            prompt="\n你：",
            assistant_prefix="\n旅遊助理：",
            exit_commands=("exit", "quit", "離開", "結束"),
        )
        # 使用者輸入離開指令後顯示結束訊息
        print("\n👋 再見")
    finally:
        # 不論正常離開或執行時發生錯誤，都關閉 npx 啟動的 MCP 子程序
        adapter.stop()


# 只有直接執行 main.py 時才啟動系統；被其他檔案 import 時不會自動執行
if __name__ == "__main__":
    main()
