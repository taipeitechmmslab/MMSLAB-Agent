from llama_index.core import (
    DocumentSummaryIndex,
    KeywordTableIndex,
    SimpleDirectoryReader,
    StorageContext,
    SummaryIndex,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter

# 每個 Chunk 約 256 個 token
CHUNK_SIZE = 256
# 前後兩個 Chunk 重疊 50 個 token，避免句子被切斷
CHUNK_OVERLAP = 50

# 定義函式 load_data_docs，讀取 data 資料夾裡所有的旅遊紀錄
def load_data_docs():
    # 設定要讀取的資料夾與檔案
    reader = SimpleDirectoryReader(
        input_dir="./data",  # 一個 .txt 檔就是一筆旅遊紀錄
        required_exts=[".txt"],  # 只讀取 .txt 檔案
    )
    # 讀取檔案，回傳 Document 列表
    return reader.load_data()

# 定義函式 build_splitter，建立四種 Index 共用的切分器
def build_splitter():
    return SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

# 定義函式 build_summary_index，建立 SummaryIndex
def build_summary_index(documents, splitter):  # 輸入 Document 列表與切分器
    print("建立 SummaryIndex...")
    # 切好 Chunk 後依序存起來，建立時不呼叫 LLM
    return SummaryIndex.from_documents(documents, transformations=[splitter])

# 定義函式 build_vector_index，建立 VectorStoreIndex，並把向量存進 Milvus 向量資料庫
def build_vector_index(documents, splitter, embed_model, vector_store):  # 輸入資料與模型
    print("建立 VectorStoreIndex + Milvus...")
    # 指定向量要存進 Milvus 向量資料庫
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,  # 存進 Milvus 向量資料庫
        transformations=[splitter],  # 先切成 Chunk
        embed_model=embed_model,  # 把每個 Chunk 轉成向量
    )

# 定義函式 build_document_summary_index，建立 DocumentSummaryIndex
def build_document_summary_index(documents, splitter, llm, embed_model):  # 輸入資料
    print("建立 DocumentSummaryIndex（每篇各生一段 LLM 摘要）...")
    return DocumentSummaryIndex.from_documents(
        documents,
        llm=llm,  # 為每一篇旅遊紀錄產生摘要
        embed_model=embed_model,  # 把摘要轉成向量，查詢時用來挑出相關的旅遊紀錄
        transformations=[splitter],  # 使用相同的切分設定
        show_progress=True,  # 顯示建立進度
    )

# 定義函式 build_keyword_index，建立 KeywordTableIndex
def build_keyword_index(documents, splitter, llm):  # 輸入 Document 列表、切分器與 LLM
    print("建立 KeywordTableIndex（LLM 抽關鍵字）...")
    return KeywordTableIndex.from_documents(
        documents,
        llm=llm,  # 從每個 Chunk 抽出關鍵字
        transformations=[splitter],  # 使用相同的切分設定
        show_progress=True,  # 顯示建立進度
    )
