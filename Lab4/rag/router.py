import logging
from llama_index.core import PromptTemplate
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import PydanticSingleSelector
from llama_index.core.tools import QueryEngineTool
from . import clients, indexes

# VectorStoreIndex 取回最相似的 5 個 Chunk
VECTOR_TOP_K = 5
# DocumentSummaryIndex 挑出最相關的 3 篇旅遊紀錄
DOC_SUMMARY_TOP_K = 3
# KeywordTableIndex 最多取回 5 個 Chunk
KEYWORD_TOP_K = 5

# 自訂 Prompt，讓 RAG 整理過往的台灣旅遊經驗，不直接回答問題
ORGANIZE_QA_TEMPLATE = PromptTemplate(
    "以下是從使用者過往「台灣」旅遊紀錄中檢索到的片段：\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "使用者本次提問：{query_str}\n\n"
    "請從上述片段中整理出與本次提問『主題或感受』相關的『過往台灣經驗』，"
    "包括地點、體驗、感受、評價，用 2-4 句話歸納"
    "（若提問涉及花費、預算或統計，改為逐筆列出每筆紀錄的金額與天數，不受 2-4 句限制）。\n"
    "規則：\n"
    "1. 不要直接回答問題本身（例如不要說『京都有/沒有 X』）；"
    "但若提問涉及花費、預算、金額或統計，必須完整保留並逐筆列出片段中的"
    "金額、天數與「每人／總花費」標記，不得省略或概括成定性描述\n"
    "2. 不要捏造未在片段中出現的內容\n"
    "3. 若片段中完全無相關經驗，直接回「無相關過往經驗」\n"
    "4. 使用繁體中文輸出\n"
    "整理結果："
)

# 定義函式 _build_tools，把四種 Index 包成 RouterQueryEngine 可以選擇的工具
def _build_tools(summary_index, vector_index, doc_summary_index, keyword_index, llm, summary_llm):  # 輸入 Index 與 LLM
    # 把 SummaryIndex 包成工具，description 寫明適合總覽型問題
    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_index.as_query_engine(
            llm=summary_llm,  # 查詢時會讀過全部 Chunk，所以使用摘要 LLM
            response_mode="tree_summarize",  # 先分組摘要，再逐層合併成總結
            summary_template=ORGANIZE_QA_TEMPLATE,
        ),
        description=(
            "適合回答文件整體內容、跨文件摘要、主題總覽與綜合分析等總覽型問題，"
            "如歸納使用者整體旅遊偏好、旅行風格，或跨紀錄的統計與聚合（平均花費、預算結構）。"
            "例如：歸納我過去旅行偏好的行程節奏，安排日本行程時一天排幾個景點比較適合我。"
        ),
    )

    # 把 VectorStoreIndex 包成工具，description 寫明適合語意型問題
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_index.as_query_engine(
            llm=llm,
            similarity_top_k=VECTOR_TOP_K,
            text_qa_template=ORGANIZE_QA_TEMPLATE,
        ),
        description=(
            "適合回答景點體驗、美食評價、住宿細節、行程安排等針對單一片段的語意型問題，"
            "以語意相似度檢索，問題與紀錄用詞不同也能命中，重點是意思相近。"
            "例如：日本有沒有類似我在花蓮走過那條沿溪步道的健行路線。"
        ),
    )

    # 把 DocumentSummaryIndex 包成工具，description 寫明適合文件摘要篩選型問題
    doc_summary_tool = QueryEngineTool.from_defaults(
        query_engine=doc_summary_index.as_query_engine(
            llm=llm,  # 使用主要 LLM 整理回答
            retriever_mode="llm",  # 由 LLM 比對每篇摘要，挑出最相關的旅遊紀錄
            choice_top_k=DOC_SUMMARY_TOP_K,
            text_qa_template=ORGANIZE_QA_TEMPLATE,
        ),
        description=(
            "適合「以整趟旅行為單位」找出與提問最相關的幾筆完整旅遊紀錄，"
            "先比對每篇文件的摘要挑出最相關的旅行，再帶回那幾趟的完整內容回顧。"
            "重點：答案是『哪一趟／哪幾趟』整篇行程（天數、節奏、整體安排）；"
            "若只問單一景點或單一步道的細節，要用 VectorStoreIndex。"
            "例如：哪一趟整體規劃和我這次想要的最像、哪一趟玩得最不順我想避開。"
        ),
    )

    # 把 KeywordTableIndex 包成工具，description 寫明適合精確名詞型問題
    keyword_tool = QueryEngineTool.from_defaults(
        query_engine=keyword_index.as_query_engine(
            llm=llm,  # 使用主要 LLM 整理回答
            num_chunks_per_query=KEYWORD_TOP_K,
            text_qa_template=ORGANIZE_QA_TEMPLATE,
        ),
        description=(
            "適合回答『某個確切名稱、專有名詞是否出現、出現在哪幾趟』的精確比對問題，"
            "例如民宿名（木門厝）、店名（文章牛肉湯）、步道名（砂卡礑步道）、"
            "特有名詞（海龜、達悟族、螢火蟲）的字面命中查詢。"
            "與 VectorStoreIndex 的差別：這裡要精確字面命中，不是語意相似。"
        ),
    )

    # 回傳四個工具
    return [summary_tool, vector_tool, doc_summary_tool, keyword_tool]

# 定義函式 build_router_query_engine，建立 RouterQueryEngine
def build_router_query_engine():
    # 建立主要 LLM、摘要 LLM、Embedding Model 與 Milvus 向量資料庫連線
    llm = clients.build_llm()
    summary_llm = clients.build_summary_llm()
    embed_model = clients.build_embed_model()
    vector_store = clients.build_milvus_vector_store()
    splitter = indexes.build_splitter()

    # 讀取旅遊紀錄
    print("🔨 讀取 ./data 旅遊紀錄")
    documents = indexes.load_data_docs()

    # 建立四種 Index，DocumentSummaryIndex 與 KeywordTableIndex 使用摘要 LLM
    summary_index = indexes.build_summary_index(documents, splitter)
    vector_index = indexes.build_vector_index(documents, splitter, embed_model, vector_store)
    doc_summary_index = indexes.build_document_summary_index(documents, splitter, summary_llm, embed_model)
    keyword_index = indexes.build_keyword_index(documents, splitter, summary_llm)

    # 把四種 Index 包成工具
    tools = _build_tools(summary_index, vector_index, doc_summary_index, keyword_index, llm, summary_llm)

    # 建立 RouterQueryEngine，由 LLM 讀取每個工具的 description 後選出一個
    router_engine = RouterQueryEngine(
        selector=PydanticSingleSelector.from_defaults(llm=llm),
        query_engine_tools=tools,
        llm=llm,  # 把檢索結果整理成最後的回答
        verbose=False,  # 不顯示內建的選路訊息，改由 chat.py 顯示
    )

    # 關閉 RouterQueryEngine 內部的選路紀錄，避免重複顯示
    logging.getLogger("llama_index.core.query_engine.router_query_engine").setLevel(
        logging.WARNING
    )

    # 提示建立完成，並回傳 RouterQueryEngine
    print("✅ RouterQueryEngine 建立完成（SummaryIndex + VectorStoreIndex + DocumentSummaryIndex + KeywordTableIndex）")
    return router_engine
