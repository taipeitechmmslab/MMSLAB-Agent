import os
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.milvus import MilvusVectorStore

# NVIDIA NIM 的 API 端點
LLM_API_BASE = "https://integrate.api.nvidia.com/v1"
# LLM 一次最多能處理的 token 數
LLM_CONTEXT_WINDOW = 128000
# 等待 LLM 回應的秒數上限
LLM_TIMEOUT = 300.0

# Milvus 向量資料庫的位址
MILVUS_URI = "http://localhost:19530"
# 存放旅遊紀錄向量的資料集名稱
MILVUS_COLLECTION = "travel_preferences"
# 向量的維度，要和 Embedding Model 的輸出一致
EMBED_DIM = 4096

# 定義函式 build_llm，透過 NVIDIA NIM 建立 LLM
def build_llm(model=None):  # 輸入要使用的模型名稱
    return OpenAILike(
        api_base=LLM_API_BASE,  # NVIDIA NIM 的 API 端點
        api_key=os.getenv("NVIDIA_API_KEY"),  # 使用 .env 設定的 API Key
        # 沒有指定模型時，使用 .env 設定的主要 LLM
        model=model or os.getenv("CHAT_MODEL"),
        is_chat_model=True,  # 使用對話模式
        is_function_calling_model=True,  # 讓 LLM 可以輸出選路結果
        context_window=LLM_CONTEXT_WINDOW,
        timeout=LLM_TIMEOUT,
    )

# 定義函式 build_summary_llm，建立回應較快的摘要 LLM
def build_summary_llm():
    # 使用 .env 設定的摘要 LLM
    return build_llm(os.getenv("SUMMARY_MODEL"))

# 定義函式 build_embed_model，建立把文字轉成向量的 Embedding Model
def build_embed_model():
    # 使用 .env 設定的 Embedding Model
    return NVIDIAEmbedding(model=os.getenv("EMBEDDING_MODEL"))

# 定義函式 build_milvus_vector_store，建立 Milvus 向量資料庫的連線
def build_milvus_vector_store():
    return MilvusVectorStore(
        uri=MILVUS_URI,  # Milvus 向量資料庫的位址
        collection_name=MILVUS_COLLECTION,  # 存放旅遊紀錄向量的資料集
        dim=EMBED_DIM,  # 向量的維度
        overwrite=True,  # 每次啟動都覆寫，讓資料和 data 資料夾一致
    )
