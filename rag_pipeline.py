# ✅ Updated for Chroma >= 0.5.4 with DuckDB backend
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from typing import List, Dict
import os
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.schema import Document
from utils.pdf_loader import load_pdf_as_documents
from config import CHROMA_DIR


# ✅ HuggingFace Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Build LLM using HuggingFace
def build_llm():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    text_gen_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512
    )

    return HuggingFacePipeline(pipeline=text_gen_pipeline)

llm = build_llm()

# ✅ Get Chroma Vectorstore with DuckDB backend
def get_vectorstore():
    if not os.path.exists(CHROMA_DIR):
        raise ValueError("❌ Chroma DB not found. Please ingest documents first.")
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model,
        client_settings={"chroma_db_impl": "duckdb+parquet"}
    )

# ✅ Ingest PDFs into Chroma DB
def ingest_filepaths(file_paths: List[str], chunk_size: int = 1000, chunk_overlap: int = 150):
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
        print(f"✅ Chroma DB index created and saved at {CHROMA_DIR}")
        return len(all_docs)
    else:
        print("⚠️ No documents added.")
        return 0

# ✅ Retriever
def get_retriever(top_k: int = 5):
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": top_k})

# ✅ Format citations
def format_citation_blocks(docs: List[Document]) -> List[Dict]:
    return [
        {
            "source": d.metadata.get("source", "unknown"),
            "page": d.metadata.get("page", "?"),
            "snippet": d.page_content[:400]
        }
        for d in docs
    ]

# ✅ Compose prompt
def _compose_prompt(question: str, contexts: List[Document]) -> str:
    context_text = "\n\n".join([
        f"[Source {i}] {c.metadata.get('source','?')}, p.{c.metadata.get('page','?')}:\n{c.page_content}"
        for i, c in enumerate(contexts, start=1)
    ])
    return (
        "You are Arc, an academic research assistant. Answer strictly from the provided sources. "
        "Include inline citations like (FileName.pdf, p. X). If you cannot find evidence, say so.\n\n"
        f"Question: {question}\n\nContext:\n{context_text}\nAnswer concisely with evidence."
    )

# ✅ Answer query
def answer_query(question: str, top_k: int = 5):
    retriever = get_retriever(top_k=top_k)
    docs = retriever.get_relevant_documents(question)
    citations = format_citation_blocks(docs)

    prompt = _compose_prompt(question, docs)
    response = llm(prompt)

    return {
        "answer": response["generated_text"] if isinstance(response, dict) else str(response),
        "citations": citations
    }

# ✅ Summarize documents
def summarize_documents(documents: List[Document], length: str = "short"):
    text_combined = "\n\n".join([d.page_content for d in documents[:20]])
    prompt = f"Summarize in {length} detail:\n{text_combined}"
    response = llm(prompt)
    return response["generated_text"] if isinstance(response, dict) else str(response)
