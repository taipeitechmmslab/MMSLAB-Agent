from datetime import date
from dotenv import load_dotenv
from crewai import Flow
from crewai.events.utils.console_formatter import ConsoleFormatter
from crewai.experimental.conversational import ConversationState
from crewai.flow import listen
from crewai_tools.adapters.tool_collection import ToolCollection
from crew import build_crew
from tools import start_mcp_tools

# 讀取 .env 裡的設定
load_dotenv()

# 定義類別 TravelFlow，保存對話歷史，並把每輪問題交給 Crew
class TravelFlow(Flow[ConversationState]):  # TravelFlow 類別繼承 Flow 父類別
    # 開啟多輪對話，讓 Flow 自動保存使用者與 Agent 的訊息
    conversational = True

    # 保存 main 函式啟動的 MCP 工具清單，讓每輪的 Crew 共用
    mcp_tools: ToolCollection

    # 定義函式 route_turn，把所有旅遊問題都交給 Crew 處理
    def route_turn(self, context: dict) -> str:  # 輸入本輪的對話內容
        # 回傳 travel，觸發下面的 handle_travel 函式
        return "travel"

    # 收到 travel 後，執行這一輪的旅遊問答
    @listen("travel")
    # 定義函式 handle_travel，建立本輪 Crew，並回傳最後的回答
    def handle_travel(self) -> str:
        # 排除本輪的問題，只取最近兩輪對話作為背景
        history = self.conversation_messages[:-1][-4:]
        # 把對話整理成文字，沒有先前對話時寫「沒有先前對話」
        conversation_context = "\n\n".join(
            f"{message['role']}：{message['content']}" for message in history
        ) or "（沒有先前對話。）"

        # 每輪重新建立 Crew，避免上一輪的狀態影響本輪
        crew = build_crew(self.mcp_tools)

        # 清除上一輪的完成事件
        ConsoleFormatter.crew_completion_printed.clear()

        # 把本輪問題、今天日期與對話背景交給 Task 執行
        result = crew.kickoff(
            inputs={
                "query": self.state.current_user_message,
                "today": date.today().isoformat(),
                "conversation_context": conversation_context,
            }
        )

        # 最多等 10 秒，避免完成訊息插入下一輪的輸入提示
        ConsoleFormatter.crew_completion_printed.wait(timeout=10)
        # 把結果轉成文字回傳，交給 Flow 保存並顯示
        return str(result)

# 定義函式 main，啟動 MCP 工具並開始多輪對話
def main() -> None:
    # 啟動四個 MCP Server，程式執行期間只啟動一次，每輪的 Crew 共用
    adapter, tools = start_mcp_tools()

    try:
        # 顯示系統名稱與三個示範問題
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

        # 建立 TravelFlow，並用 chat 開始多輪對話
        TravelFlow(
            mcp_tools=tools,  # 傳入 MCP 工具清單
            suppress_flow_events=True,  # 不顯示 Flow 本身的事件訊息
            tracing=False,  # 關閉背景追蹤
        ).chat(
            prompt="\n你：",  # 輸入提示
            assistant_prefix="\n旅遊助理：",  # 回答前顯示的名稱
            exit_commands=("exit", "quit", "離開", "結束"),  # 結束對話的指令
        )
        # 使用者結束對話後顯示再見
        print("\n再見")
    finally:
        # 不論正常結束或發生錯誤，都關閉 npx 啟動的 MCP Server
        adapter.stop()

# 執行 main 函式，啟動整個程式
if __name__ == "__main__":
    main()
