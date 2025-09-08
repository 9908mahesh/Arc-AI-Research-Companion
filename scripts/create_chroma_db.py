import os
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from utils.pdf_loader import load_pdf_as_documents
from config import CHROMA_DIR

# ✅ Initialize HuggingFace embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_chroma_index(file_paths, chunk_size=1000, chunk_overlap=150):
    docs_added = 0
    all_docs = []

    for p in file_paths:
        docs = load_pdf_as_documents(p)
        if docs:
            print(f"✅ Loaded {len(docs)} chunks from {p}")
            all_docs.extend(docs)
            docs_added += len(docs)
        else:
            print(f"⚠️ No content extracted from {p}")

    if all_docs:
        vectorstore = Chroma.from_documents(all_docs, embedding_model, persist_directory=CHROMA_DIR)
        vectorstore.persist()
        print("✅ Chroma DB index created and saved.")
    else:
        print("⚠️ No documents added to Chroma DB.")

    return docs_added

if __name__ == "__main__":
    # Example usage: replace with your PDF file paths
    sample_files = ["sample1.pdf", "sample2.pdf"]
    create_chroma_index(sample_files)
