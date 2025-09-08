from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

CHROMA_DIR = "chroma_db"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

CHROMA_SETTINGS = {
    "chroma_db_impl": "duckdb+parquet",
    "persist_directory": CHROMA_DIR
}

def create_chroma_index(file_paths, chunk_size=1000, chunk_overlap=150):
    docs_added = 0
    all_docs = []

    for p in file_paths:
        from utils.pdf_loader import load_pdf_as_documents
        docs = load_pdf_as_documents(p)
        if docs:
            print(f"✅ Loaded {len(docs)} chunks from {p}")
            all_docs.extend(docs)
            docs_added += len(docs)
        else:
            print(f"⚠️ No content extracted from {p}")

    if all_docs:
        vectorstore = Chroma.from_documents(
            all_docs,
            embedding_model,
            persist_directory=CHROMA_DIR,
            client_settings=CHROMA_SETTINGS
        )
        vectorstore.persist()
        print("✅ Chroma DB index created and saved.")
    else:
        print("⚠️ No documents added to Chroma DB.")

    return docs_added
