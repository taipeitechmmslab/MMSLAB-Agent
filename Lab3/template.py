# 定義函式 build_preference_query，把使用者的問題改寫成搜尋偏好用的文字
def build_preference_query(user_query: str) -> str:  # 輸入使用者的問題
    # 在使用者的問題後面，補上想找的偏好內容
    return (
        f"根據使用者目前的旅遊需求「{user_query}」，"
        "從之前旅行紀錄中檢索與本次行程風格相關的內容，"
        "找出使用者的旅行風格偏好，包括景點類型、住宿評價、預算分配等個人化資訊，"
        "以及下次想去的地方。"
    )
