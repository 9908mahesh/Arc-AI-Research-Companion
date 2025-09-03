from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from config import OPENAI_CHAT_MODEL, OPENAI_EMBED_MODEL, OPENAI_API_KEY, DEFAULT_TOP_K
from utils.pinecone_helper import create_index_if_not_exists, get_vectorstore
from utils.pdf_loader import load_pdf_as_documents
from typing import List, Dict

# Ensure index exists
create_index_if_not_exists(dim=1536)

def ingest_filepaths(file_paths: List[str], chunk_size:int=1000, chunk_overlap:int=150):
    """
    Ingest multiple PDFs into Pinecone vectorstore.
    Returns number of documents added.
    """
    vs = get_vectorstore()
    docs_added = 0
    for p in file_paths:
        docs = load_pdf_as_documents(p)
        if docs:
            vs.add_documents(docs)
            docs_added += len(docs)
    return docs_added

def get_retriever(top_k:int = None):
    vs = get_vectorstore()
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": top_k or DEFAULT_TOP_K})
    return retriever

def build_chat_model(temperature=0.0):
    return ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=temperature, openai_api_key=OPENAI_API_KEY)

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

def make_prompt_system():
    # system guardrails
    return (
        "You are Arc, an academic research assistant. Answer strictly from the provided sources. "
        "Always include inline citations like (FileName.pdf, p. X) and an Evidence section quoting short snippets. "
        "If the retrieved sources do not support an answer, say you cannot find evidence."
    )

def _compose_messages(question:str, contexts:List[Document]):
    system = make_prompt_system()
    context_text = ""
    for i, c in enumerate(contexts, start=1):
        meta = c.metadata or {}
        context_text += f"[Source {i}] {meta.get('source','?')}, p.{meta.get('page','?')}:\\n{c.page_content}\\n\\n"
    # ChatPromptTemplate can be used but we pass messages directly in langchain Chat API chain
    user_msg = f"Question: {question}\\n\\nContext:\\n{context_text}\\n\nAnswer concisely but include Evidence section."
    return [
        {"role":"system", "content": system},
        {"role":"user", "content": user_msg}
    ]

def answer_query(question: str, top_k:int=None, mode:str="detailed"):
    """
    Modes: 'brief' (short answer), 'detailed' (longer), 'citations' (return only citations), 'summary' (structured summary)
    """
    retriever = get_retriever(top_k=top_k)
    docs = retriever.get_relevant_documents(question)
    citations = format_citation_blocks(docs)

    # Build messages with retrieved contexts
    messages = _compose_messages(question, docs)

    # choose temperature and length based on mode
    if mode == "brief":
        temp = 0.0
    elif mode == "citations":
        # we will return only citations, still call model for formatting
        temp = 0.0
    else:
        temp = 0.2

    llm = build_chat_model(temperature=temp)

    # Use RetrievalQA chain for convenience: it will use retriever internally; but we want to pass contexts for citation control.
    # We'll call the llm chat model directly for more control.
    resp = llm.apredict(messages=messages) if hasattr(llm, "apredict") else llm(messages=messages)
    # llm returns an LLMResult or string depending on langchain version
    if isinstance(resp, dict) and "content" in resp:
        answer_text = resp["content"]
    else:
        # in many langchain versions ChatOpenAI returns a ChatGeneration with .content nested
        try:
            answer_text = resp[0].message.content
        except Exception:
            answer_text = str(resp)

    # Build return payload
    return {
        "answer": answer_text,
        "citations": citations
    }

def summarize_documents(documents: List[Document], length: str = "short"):
    """
    length: 'short' (~3 bullets), 'medium' (~6 bullets), 'long' (~200-400 words)
    """
    text_combined = "\\n\\n".join([d.page_content for d in documents[:20]])  # limit to first 20 chunks to control cost
    prompt = f"Produce a {length} structured summary (Background, Methods, Results, Limitations, Key takeaways) strictly using the following extracted content:\\n\\n{text_combined}"
    llm = build_chat_model(temperature=0.2)
    messages = [{"role":"system", "content": make_prompt_system()}, {"role":"user", "content": prompt}]
    resp = llm(messages=messages)
    try:
        summary = resp[0].message.content
    except Exception:
        summary = str(resp)
    return summary
