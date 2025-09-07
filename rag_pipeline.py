from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.docstore.document import Document
from typing import List, Dict
from config import OPENAI_CHAT_MODEL, OPENAI_EMBED_MODEL, OPENAI_API_KEY, DEFAULT_TOP_K
from utils.pinecone_helper import get_vectorstore, create_index_if_not_exists
from utils.pdf_loader import load_pdf_as_documents

# ✅ Ensure Pinecone index exists before any ingestion or retrieval
create_index_if_not_exists(dim=1536)

# ✅ Updated ingest_filepaths with debug logs
def ingest_filepaths(file_paths: List[str], chunk_size: int = 1000, chunk_overlap: int = 150):
    """
    Ingest multiple PDFs into Pinecone vectorstore.
    Returns number of documents added.
    """
    vs = get_vectorstore()
    docs_added = 0

    for p in file_paths:
        docs = load_pdf_as_documents(p)
        if docs:
            print(f"✅ Loaded {len(docs)} chunks from {p}")
            try:
                vs.add_documents(docs)
                print(f"✅ Successfully ingested {len(docs)} chunks from {p} into Pinecone")
                docs_added += len(docs)
            except Exception as e:
                print(f"❌ Error adding documents from {p}: {str(e)}")
        else:
            print(f"⚠️ No content extracted from {p}")

    return docs_added

# ✅ Retriever function for similarity search
def get_retriever(top_k: int = None):
    vs = get_vectorstore()
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": top_k or DEFAULT_TOP_K})
    return retriever

# ✅ Build ChatOpenAI model
def build_chat_model(temperature=0.0):
    return ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=temperature, openai_api_key=OPENAI_API_KEY)

# ✅ Format citations for UI
def format_citation_blocks(docs: List[Document]) -> List[Dict]:
    out = []
    for d in docs:
        md = d.metadata or {}
        out.append({
            "source": md.get("source", "unknown"),
            "page": md.get("page", "?"),
            "snippet": d.page_content[:400],
            "score": getattr(d, "score", None)
        })
    return out

# ✅ Prompt system for Arc AI
def make_prompt_system():
    return (
        "You are Arc, an academic research assistant. Answer strictly from the provided sources. "
        "Always include inline citations like (FileName.pdf, p. X) and an Evidence section quoting short snippets. "
        "If the retrieved sources do not support an answer, say you cannot find evidence."
    )

# ✅ Compose messages for LLM
def _compose_messages(question: str, contexts: List[Document]):
    system = make_prompt_system()
    context_text = ""
    for i, c in enumerate(contexts, start=1):
        meta = c.metadata or {}
        context_text += f"[Source {i}] {meta.get('source','?')}, p.{meta.get('page','?')}:\n{c.page_content}\n\n"
    user_msg = f"Question: {question}\n\nContext:\n{context_text}\n\nAnswer concisely but include Evidence section."
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg}
    ]

# ✅ Answer query using retrieved documents
def answer_query(question: str, top_k: int = None, mode: str = "detailed"):
    retriever = get_retriever(top_k=top_k)
    docs = retriever.get_relevant_documents(question)
    citations = format_citation_blocks(docs)

    messages = _compose_messages(question, docs)

    temp = 0.0 if mode in ["brief", "citations"] else 0.2

    llm = build_chat_model(temperature=temp)
    resp = llm(messages=messages)

    try:
        answer_text = resp[0].message.content
    except Exception:
        answer_text = str(resp)

    return {
        "answer": answer_text,
        "citations": citations
    }

# ✅ Summarize documents
def summarize_documents(documents: List[Document], length: str = "short"):
    text_combined = "\n\n".join([d.page_content for d in documents[:20]])
    prompt = f"Produce a {length} structured summary (Background, Methods, Results, Limitations, Key takeaways) strictly using the following extracted content:\n\n{text_combined}"
    llm = build_chat_model(temperature=0.2)
    messages = [{"role": "system", "content": make_prompt_system()}, {"role": "user", "content": prompt}]
    resp = llm(messages=messages)

    try:
        summary = resp[0].message.content
    except Exception:
        summary = str(resp)

    return summary
