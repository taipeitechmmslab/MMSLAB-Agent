from datetime import date
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware
from langchain_core.messages import HumanMessage
from openai import APIError
from state import TravelState

# 定義函式 build_system_prompt，建立 executor node 的系統 Prompt
def build_system_prompt() -> str:
    return """\
你是個人化旅遊規劃的執行助理，手上有三類工具：
Tavily 搜尋（景點/住宿/交通/簽證即時資訊）、天氣查詢（Open-Meteo）、匯率換算（Frankfurter）。

工作方式：
- 依使用者訊息中給定的「執行計畫」逐步進行，自主決定該呼叫哪個工具。
- 使用者訊息附有「偏好檔案」，所有安排都必須對齊它：Tavily 搜尋關鍵詞要由偏好推導
  （例：偏好在地文創→搜「大阪 在地 文創 街區 巷弄美食」），「禁止」用「必去景點」「熱門景點」
  這類與偏好無關的通用詞；偏好檔案明確不喜歡的類型（如過度觀光化商圈）不可當主力行程，
  頂多以「順路短暫停留」帶過並說明理由。
- 選點優先序：先依偏好檔案與需求選定景點，票券（周遊卡/N日券）只是「事後」的省錢工具——
  景點選完才評估買票券划不划算，「禁止」反過來為了用滿票券，把卡片涵蓋清單裡的觀光景點
  加進行程；票券不划算就老實建議單買門票，不要硬湊。
- 天氣查詢若超出預設範圍，帶入 start_date / end_date 重試。
- Tavily 搜尋一律用 search_depth='basic'、max_results 不超過 3，避免回傳過長內容撐爆 token 上限。
- 產出第一版草案前，務必對「主要付費景點門票」與「行程用到的票券（如周遊卡/一日券）效期」用官方頁查證，
  不要只憑部落格或記憶填數字。尤其推薦任何有使用期限的票券前，必須先確認它在旅遊日期當天仍販售、仍有效，
  否則不要推薦、也不要把行程的免費入場建立在它上面。
- 資訊蒐集完成後，直接產出完整的每日行程草案（不要只回工具結果）。

資料正確性原則（修訂時尤其重要）：
- 計畫裡若出現具體數字的修正要求（例如「把票價改成 X」），那是上一輪審核的轉述，「不是」最終事實。
  請你自己用工具（優先查官方網站/官方票務頁）查證後，「以官方來源為準」，不要盲目照抄計畫給的數字。
- 同一事實若多個來源數字不同（官網 vs 比價站 vs 部落格），一律採信權威性最高者：
  官方網站／官方票務頁 > 旅遊比價站 > 個人部落格/論壇。採用官方值，並可在該行簡短註明「(官方價)」。
- 若你查到的官方數據與計畫轉述的數字衝突，以官方數據為準，無須反覆改動。
- 查不到就誠實標提示，不要假稱官方：某個門票/票價你查了一輪仍找不到可靠官方來源時，
  「禁止」宣稱「官方確認」或編一個數字硬寫上去。請用合理估計值並在該行標「(門票以官網/現場為準)」，
  當成提示帶過即可，不要在後續每輪反覆糾結同一個查不到的數字。

修訂時的最小改動原則（避免越改越亂）：
- 只動上一輪審核 issues「明確點到」的地方（改那個數字、修那段時間衝突、補那個備案）。
- 沒被點到、且上一版已經合理的行程段落，「原樣保留」，不要整天重排、不要換掉沒問題的景點，
  以免修好一處又在別處製造新的時間或動線矛盾。
- 修正數字（門票、價格、開放時間）時「直接改原本那一行」，不要為了交代修正而「新增一行價格摘要或合計」
  當成行程項目塞進時間表——同一景點的票價只出現在它自己那一行，預算合計只放在最後的預算段落。

行程草案內容：
1. 先以三段式說明偏好：原文依據（引用偏好檔案中的旅行紀錄原文）／推理結果／行程影響，
   最多 3 段、每段 1-2 句。
2. 每日行程含上午、午餐、下午、晚上，每行格式「時間 景點（當地費用 / 台幣）交通」一行寫完。
   範例：・09:00 大阪城天守閣（1,200日圓 / 約260台幣）地鐵谷町四丁目站步行10分
3. 雨天景點要有室內備案。
4. 最後列預算總計（以實際匯率換算）、住宿推薦、交通建議、注意事項，合計不超過 200 字。

輸出規則：
- 整體控制在 1000 字以內，純文字，禁止任何 markdown 語法（# ## ** * ` 表格 ---- > emoji）。
- 條列用「・」或「1. 2. 3.」，不要用「-」或「*」開頭。
- 不要寫景點歷史背景、文化典故，只列實用資訊。
- 若收到修正要求，務必逐項處理問題（改數字、換地點、補備案），並在最後加「修正摘要」逐項說明「問題→如何改」。"""

# 定義函式 create_executor，建立 executor node 並回傳
def create_executor(llm, tools):  # 輸入 LLM 與工具清單
    # 建立 Agent，它會自己決定要呼叫哪些工具
    agent = create_agent(
        llm,
        tools,
        system_prompt=build_system_prompt(),
        # LLM 呼叫失敗時自動重試
        middleware=[
            ModelRetryMiddleware(
                # 最多重試 2 次
                max_retries=2,
                # 第一次等 5 秒
                initial_delay=5.0,
                # 之後每次等待時間加倍
                backoff_factor=2.0,
                # 最多等 15 秒
                max_delay=15.0,
                # 不加入隨機延遲
                jitter=False,
                # 只有 API 出錯才重試
                retry_on=(APIError,),
                # 全部失敗就交給 chat.py 處理
                on_failure="error",
            )
        ],
    )

    # 定義 node executor，照著計畫查資料，排出每日行程草案
    async def executor(state: TravelState) -> dict:  # 以 asynchronous 方式執行，輸入 TravelState 讀取 messages、preferences 與 plan 欄位
        # 把計畫的步驟加上編號
        plan_text = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(state["plan"]))
        # 把今天日期、偏好與計畫整理成這輪的指令
        directive = HumanMessage(content=(
            f"[今天日期] {date.today().isoformat()}（請以此推算「下週二」等相對日期，"
            "並相信天氣/匯率工具回傳的年份）\n\n"
            f"[使用者偏好檔案]\n{state.get('preferences') or '（無）'}\n\n"
            f"[執行計畫]\n{plan_text}\n\n"
            "請依上述計畫與偏好檔案呼叫工具蒐集資訊，並產出完整的每日行程草案。"
        ))
        # 把指令接在對話歷史後面
        input_messages = list(state["messages"]) + [directive]
        # 交給背景的 Agent 照計畫查資料並寫出行程草案，前景以 await 取回結果
        result = await agent.ainvoke({"messages": input_messages})

        # 只取出 Agent 這輪新產生的訊息
        new_messages = result["messages"][len(input_messages):]
        # 新訊息會自動接在對話歷史後面
        return {"messages": new_messages}

    # 回傳這個 node，交給 main.py 使用
    return executor
