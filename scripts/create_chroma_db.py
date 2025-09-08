import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from utils.pdf_loader import load_pdf_as_documents
from config import CHROMA_DIR

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

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
            persist_directory=CHROMA_DIR,
            client_settings={"chroma_db_impl": "duckdb+parquet"}
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
