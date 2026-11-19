from llama_index.tools.mcp import BasicMCPClient, McpToolSpec

# 定義函式 load_mcp_tools，啟動 MCP 工具並取得工具清單
async def load_mcp_tools():
    # 建立 Tavily 的 MCP Client，用來上網搜尋即時資訊
    tavily_client = BasicMCPClient("npx", args=["-y", "tavily-mcp@latest"])
    # 轉成 Agent 可以使用的工具
    tavily_spec = McpToolSpec(client=tavily_client)
    # 等 Tavily 回應，取得工具清單
    tavily_tools = await tavily_spec.to_tool_list_async()

    # 建立 Open-Meteo 的 MCP Client，用來查詢天氣，不需要 API Key
    meteo_client = BasicMCPClient("npx", args=["-y", "open-meteo-mcp-server"])
    # 轉成 Agent 可以使用的工具
    meteo_spec = McpToolSpec(client=meteo_client)
    # 等 Open-Meteo 回應，取得工具清單
    meteo_tools = await meteo_spec.to_tool_list_async()

    # 顯示載入的工具數量，並把兩份工具清單合併回傳
    print(f"🔧 已載入 MCP 工具：Tavily {len(tavily_tools)} 個、Open-Meteo {len(meteo_tools)} 個")
    return tavily_tools + meteo_tools
