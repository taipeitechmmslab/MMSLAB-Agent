# 定義函式 read_query，讀取使用者輸入的問題
def read_query(turn: int) -> str | None:  # 輸入目前是第幾輪對話
    try:
        # 顯示第幾輪，讀取使用者輸入，並去掉前後的空白
        query = input(f"[第 {turn} 輪] 你：").strip()
    except (EOFError, KeyboardInterrupt):
        # 使用者按 Ctrl+C 或 Ctrl+D，就結束對話
        return None

    # 使用者輸入 exit 或 quit，也結束對話
    if query.lower() in {"exit", "quit"}:
        return None

    # 回傳使用者輸入的問題
    return query
