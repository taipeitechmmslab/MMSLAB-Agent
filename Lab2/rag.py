import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_milvus import Milvus
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 定義函式 load_data_docs，讀取 data 資料夾裡所有的旅遊紀錄
def load_data_docs():
    # 設定要讀取的資料夾與檔案
    loader = DirectoryLoader(
        "./data",
        glob="**/*.txt",  # 讀取所有 .txt 檔案
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},  # 用 UTF-8 讀取，避免中文亂碼
    )
    # 讀取檔案，回傳 Document 列表
    return loader.load()

# 定義函式 split_docs，將較長的旅行紀錄切成適合向量檢索的 Chunk
def split_docs(documents):  # 輸入 Document 列表
    # 設定 Chunk 的大小與重疊長度
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=256,  # 每個 Chunk 最多 256 個字
        chunk_overlap=50,  # 前後兩個 Chunk 重疊 50 個字，避免句子被切斷
        separators=["\n\n", "\n", "。", "，", " ", ""],  # 優先在換行、句號、逗號的地方切開
    )

    # 切分所有 Document
    chunks = splitter.split_documents(documents)
    # 回傳切好的 Chunk
    return chunks

# 定義函式 build_vector_store，把每個 Chunk 轉成向量，存進 Milvus 向量資料庫
def build_vector_store():
    # 使用 .env 設定的 Embedding Model
    embeddings = NVIDIAEmbeddings(
        model=os.getenv("EMBEDDING_MODEL"),
    )

    # 提示使用者正在建立資料庫
    print("建立 Milvus collection，讀取 ./data")

    # 讀取旅遊紀錄，並切成 Chunk
    documents = load_data_docs()
    chunks = split_docs(documents)

    # 把每個 Chunk 轉成向量，存進 Milvus 向量資料庫
    vector_store = Milvus.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="travel_preferences",  # 存放旅遊偏好的資料集名稱
        connection_args={"uri": "http://localhost:19530"},  # Milvus 向量資料庫的位址
        drop_old=True,  # 每次啟動都先清掉舊資料
    )
    # 顯示存了幾筆資料
    print(f"已建立 {len(chunks)} 筆向量，存入 http://localhost:19530")

    # 回傳建立好的向量資料庫
    return vector_store

# 定義函式 build_retriever，建立 RAG 檢索器
def build_retriever():
    # 建立 Milvus 向量資料庫
    vector_store = build_vector_store()

    # 建立 RAG 檢索器，每次找出最相關的 5 個 Chunk
    return vector_store.as_retriever(search_kwargs={"k": 5})
