# ✅ Fix for SQLite issue
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

import os
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from utils.pdf_loader import load_pdf_as_documents
from config import CHROMA_DIR

# ✅ HuggingFace Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Create Chroma index from PDFs using the new API
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
        vectorstore = Chroma.from_documents(
            documents=all_docs,
            embedding=embedding_model,
            persist_directory=CHROMA_DIR  # ✅ Directly persist with new API
        )
        vectorstore.persist()
        print(f"✅ Chroma DB index created and saved at {CHROMA_DIR}")
        return len(all_docs)
    else:
        print("⚠️ No documents added to Chroma DB.")
        return 0

# ✅ Load Chroma Vectorstore
def get_vectorstore():
    if not os.path.exists(CHROMA_DIR):
        raise ValueError("❌ Chroma DB not found. Please ingest documents first.")
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model
    )

if __name__ == "__main__":
    sample_files = ["sample1.pdf", "sample2.pdf"]
    create_chroma_index(sample_files)
