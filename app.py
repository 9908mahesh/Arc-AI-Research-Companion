import streamlit as st
import tempfile, os, time
from config import ensure_keys, OPENAI_API_KEY, OPENAI_CHAT_MODEL, OPENAI_EMBED_MODEL, DEFAULT_TOP_K, PINECONE_INDEX_NAME
from rag_pipeline import ingest_filepaths, answer_query, get_retriever, summarize_documents
from utils.pdf_loader import load_pdf_as_documents
from utils.pinecone_helper import create_index_if_not_exists
from utils.pdf_exporter import create_summary_pdf
from utils.ui_helpers import style_app, sidebar_instructions

# --- Page config & styles
st.set_page_config(page_title="Arc – AI Research Companion", layout="wide", initial_sidebar_state="expanded")
style_app()

st.title("Arc — AI Research Companion")
st.markdown("**Private & powerful research assistance — cloud RAG (OpenAI + Pinecone)**")

with st.sidebar:
    sidebar_instructions()
    st.markdown("---")
    st.write("**Index:**", PINECONE_INDEX_NAME)
    if st.button("Create Pinecone Index (if missing)"):
        try:
            created = create_index_if_not_exists(dim=1536)
            if created:
                st.success("Index created.")
            else:
                st.info("Index already exists.")
        except Exception as e:
            st.error(f"Index creation failed: {e}")

# Ingest section
st.header("1) Upload & Ingest PDFs")
uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
chunk_size = st.number_input("Chunk size (chars)", value=1000, min_value=500, max_value=3000)
chunk_overlap = st.number_input("Chunk overlap (chars)", value=150, min_value=0, max_value=1000)

if st.button("📥 Ingest uploaded PDFs"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF before ingesting.")
    else:
        with st.spinner("Saving & ingesting PDFs..."):
            temps = []
            for f in uploaded_files:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                tmp.write(f.read())
                tmp.flush()
                tmp.close()
                temps.append(tmp.name)
            try:
                total = ingest_filepaths(temps, chunk_size, chunk_overlap)
                st.success(f"Ingested {total} chunks into index.")
            except Exception as ex:
                st.error(f"Ingest failed: {ex}")
            finally:
                for t in temps:
                    try:
                        os.unlink(t)
                    except:
                        pass

st.markdown("---")
st.header("2) Ask Questions & Get Cited Answers")

col1, col2 = st.columns([3,1])

with col1:
    mode = st.selectbox("Answer mode", options=["detailed", "brief", "citations"], index=0, help="detailed = longer answer; brief = short; citations = show retrieved sources only")
    top_k = st.slider("Retrieval Top-K", 3, 12, DEFAULT_TOP_K)
    question = st.text_input("Ask a question about the ingested corpus or uploaded PDFs:")

    if st.button("✨ Get Answer"):
        if not question.strip():
            st.warning("Please type a question.")
        else:
            with st.spinner("Retrieving & generating..."):
                try:
                    result = answer_query(question, top_k=top_k, mode=mode)
                    st.markdown("### ✅ Answer")
                    st.write(result["answer"])

                    st.markdown("### 📚 Retrieved Evidence")
                    citations = result.get("citations", [])
                    for i, c in enumerate(citations, start=1):
                        st.write(f"**{i}.** {c['source']} — p.{c['page']} (score: {c.get('score')})")
                        st.write(f"> {c['snippet'][:800]}...")
                except Exception as e:
                    st.error(f"Query failed: {e}")

with col2:
    st.markdown("### Controls")
    if st.button("🧾 Summarize entire corpus (medium)"):
        with st.spinner("Generating summary..."):
            retriever = get_retriever(top_k=DEFAULT_TOP_K)
            docs = retriever.get_relevant_documents(" ")  # simple way to fetch many docs; implementation detail
            summary = summarize_documents(docs, length="medium")
            st.markdown("### Summary")
            st.write(summary)
            # prepare PDF export
            if st.button("⬇️ Download Summary as PDF"):
                pdf_bytes = create_summary_pdf("Arc Summary", summary, [{"source":d.metadata.get("source",""), "page":d.metadata.get("page",""), "snippet":d.page_content[:200]} for d in docs[:6]])
                st.download_button("Download PDF", data=pdf_bytes, file_name="arc_summary.pdf", mime="application/pdf")

st.markdown("---")
st.caption("Arc • RAG over PDFs • Powered by OpenAI + Pinecone")
