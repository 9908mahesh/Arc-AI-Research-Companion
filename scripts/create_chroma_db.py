# ✅ Fix for SQLite issue
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

import os
import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from utils.pdf_loader import load_pdf_as_documents
from config import CHROMA_DIR

# ✅ HuggingFace Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Initialize Chroma with DuckDB
def get_chroma_client():
    return chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(chroma_db_impl="duckdb+parquet")
    )

# ✅ Create Chroma index from PDFs
def create_chroma_index(file_paths, chunk_size=1000, chunk_overlap=150):
    all_docs = []
    for p in file_paths:
        docs = load_pdf_as_documents(p)
        if docs:
            print(f"✅ Loaded {len(docs)} chunks from {p}")
            all_docs.extend(docs)
        else:
            print(f"⚠️ No content extracted from {p}")

    if all_docs:
        client = get_chroma_client()
        vectorstore = Chroma.from_documents(
            documents=all_docs,
            embedding=embedding_model,
            client=client,  # ✅ Use DuckDB
            persist_directory=CHROMA_DIR
        )
        vectorstore.persist()
        print(f"✅ Chroma DB index created and saved at {CHROMA_DIR}")
        return len(all_docs)
    else:
        print("⚠️ No documents added to Chroma DB.")
        return 0

if __name__ == "__main__":
    sample_files = ["sample1.pdf", "sample2.pdf"]
    create_chroma_index(sample_files)
