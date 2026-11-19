import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_milvus import Milvus
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 定義函式 build_retriever，把旅遊紀錄存進 Milvus 向量資料庫，並建立檢索器
def build_retriever():
    # 提示使用者正在建立資料庫
    print("🔨 建立 Milvus collection，讀取 ./data")

    # 讀取 data 資料夾裡所有的旅遊紀錄
    documents = DirectoryLoader(
        "./data",
        glob="**/*.txt",  # 讀取所有 .txt 檔案
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},  # 用 UTF-8 讀取，避免中文亂碼
    ).load()

    # 將較長的旅行紀錄切成適合向量檢索的文字片段
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=256,  # 每段最多 256 個字
        chunk_overlap=50,  # 前後兩段重疊 50 個字，避免句子被切斷
        separators=["\n\n", "\n", "。", "，", " ", ""],  # 優先在換行、句號、逗號的地方切開
    ).split_documents(documents)

    # 把每一段轉成向量，存進 Milvus 向量資料庫
    vector_store = Milvus.from_documents(
        documents=chunks,
        embedding=NVIDIAEmbeddings(model=os.getenv("EMBEDDING_MODEL")),  # 使用 .env 設定的 Embedding Model
        collection_name="travel_preferences",  # 存放旅遊偏好的資料集名稱
        connection_args={"uri": "http://localhost:19530"},  # Milvus 向量資料庫的位址
        drop_old=True,  # 每次啟動都先清掉舊資料
    )
    # 顯示存了幾筆資料
    print(f"✅ 已建立 {len(chunks)} 筆向量，存入 http://localhost:19530")

    # 建立檢索器，每次找出最相關的 5 段
    return vector_store.as_retriever(search_kwargs={"k": 5})
