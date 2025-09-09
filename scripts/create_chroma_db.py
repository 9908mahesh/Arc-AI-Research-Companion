# ✅ create_chroma_db.py - Chroma ingestion script
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
from utils.pdf_loader import load_pdf_as_documents
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import CHROMA_DIR

# ✅ HuggingFace Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_chroma_db(pdf_folder: str):
    all_docs = []
    for file_name in os.listdir(pdf_folder):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(pdf_folder, file_name)
            docs = load_pdf_as_documents(file_path)
            if docs:
                print(f"✅ Loaded {len(docs)} chunks from {file_name}")
                all_docs.extend(docs)

    if all_docs:
        vectorstore = Chroma.from_documents(
            documents=all_docs,
            embedding=embedding_model,
            persist_directory=CHROMA_DIR,
            client_settings={"chroma_db_impl": "duckdb+parquet"}
        )
        print(f"✅ Chroma DB successfully created at {CHROMA_DIR}")
    else:
        print("⚠️ No documents found for ingestion.")
