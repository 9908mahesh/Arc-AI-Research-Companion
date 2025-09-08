from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from typing import List, Dict
from utils.pdf_loader import load_pdf_as_documents
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from chromadb.config import Settings

# ✅ HuggingFace Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Chroma DB persistent directory
CHROMA_DIR = "chroma_db"
CHROMA_SETTINGS = Settings(chroma_db_impl="duckdb+parquet")  # ✅ Use DuckDB instead of SQLite
vectorstore = None

# ✅ Initialize HuggingFace LLM
def build_llm():
    model_name = "google/flan-t5-small"  # Lightweight model for Streamlit Cloud
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    text_gen_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        model_kwargs={"max_length": 512}
    )

    return HuggingFacePipeline(pipeline=text_gen_pipeline)

llm = build_llm()

# ✅ Ingest PDFs into Chroma using DuckDB
def ingest_filepaths(file_paths: List[str], chunk_size: int = 1000, chunk_overlap: int = 150):
    global vectorstore
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
        vectorstore = Chroma.from_documents(
            all_docs,
            embedding_model,
            persist_directory=CHROMA_DIR,
            client_settings=CHROMA_SETTINGS  # ✅ Use DuckDB backend
        )
        vectorstore.persist()
        print("✅ Chroma DB index created and saved (DuckDB).")

    return docs_added

# ✅ Retriever
def get_retriever(top_k: int = 5):
    global vectorstore
    if vectorstore is None:
        if os.path.exists(CHROMA_DIR):
            vectorstore = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=embedding_model,
                client_settings=CHROMA_SETTINGS  # ✅ Use DuckDB backend
