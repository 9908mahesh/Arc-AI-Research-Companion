from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from typing import List, Dict
from utils.pdf_loader import load_pdf_as_documents
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ✅ HuggingFace Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Initialize FAISS vectorstore globally (in-memory for now)
vectorstore = None

# ✅ Build HuggingFace LLM (choose small model for CPU if no GPU)
def build_llm():
    model_name = "tiiuae/falcon-7b-instruct"  # You can switch to "google/flan-t5-small" for CPU
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    text_gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=512)
    return HuggingFacePipeline(pipeline=text_gen_pipeline)

llm = build_llm()

# ✅ Ingest PDFs into FAISS
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
        vectorstore = FAISS.from_documents(all_docs, embedding_model)
        print("✅ FAISS index created and populated.")

    return docs_added

# ✅ Retriever
def get_retriever(top_k: int = 5):
    if vectorstore is None:
        raise ValueError("FAISS vectorstore is empty. Please ingest documents first.")
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})

# ✅ Format citations
def format_citation_blocks(docs: List[Document]) -> List[Dict]:
    out = []
    for d in docs:
        md = d.metadata or {}
        out.append({
            "source": md.get("source", "unknown"),
            "page": md.get("page", "?"),
            "snippet": d.page_content[:400]
        })
    return out

# ✅ Prompt system
def make_prompt_system():
    return (
        "You are Arc, an academic research assistant. Answer strictly from the provided sources. "
        "Include inline citations like (FileName.pdf, p. X). If you cannot find evidence, say so."
    )

# ✅ Compose messages
def _compose_prompt(question: str, contexts: List[Document]) -> str:
    context_text = ""
    for i, c in enumerate(contexts, start=1):
        meta = c.metadata or {}
        context_text += f"[Source {i}] {meta.get('source','?')}, p.{meta.get('page','?')}:\n{c.page_content}\n\n"
    return f"{make_prompt_system()}\n\nQuestion: {question}\n\nContext:\n{context_text}\nAnswer concisely with evidence."

# ✅ Answer query
def answer_query(question: str, top_k: int = 5):
    retriever = get_retriever(top_k=top_k)
    docs = retriever.get_relevant_documents(question)
    citations = format_citation_blocks(docs)

    prompt = _compose_prompt(question, docs)
    response = llm(prompt)

    return {
        "answer": response,
        "citations": citations
    }

# ✅ Summarize documents
def summarize_documents(documents: List[Document], length: str = "short"):
    text_combined = "\n\n".join([d.page_content for d in documents[:20]])
    prompt = f"Summarize in {length} detail:\n{text_combined}"
    summary = llm(prompt)
    return summary
