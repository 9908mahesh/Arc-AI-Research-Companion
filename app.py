import streamlit as st
import tempfile
import os
from rag_pipeline import ingest_filepaths, answer_query, get_retriever, summarize_documents
from utils.pdf_exporter import create_summary_pdf
from utils.ui_helpers import style_app, sidebar_instructions

# --- Page Config & Styles ---
st.set_page_config(page_title="Arc – AI Research Companion", layout="wide", initial_sidebar_state="expanded")
style_app()

# --- Title ---
st.title("📚 Arc — AI Research Companion")
st.markdown("**Private & powerful research assistant — RAG using HuggingFace + ChromaDB**")

# --- Sidebar ---
with st.sidebar:
    sidebar_instructions()
    st.markdown("---")
    st.write("**Current Vector DB:** `Chroma (Local)`")

# --- Section 1: PDF Ingestion ---
st.header("1) Upload & Ingest PDFs")
uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
chunk_size = st.number_input("Chunk size (characters)", value=1000, min_value=500, max_value=3000)
chunk_overlap = st.number_input("Chunk overlap (characters)", value=150, min_value=0, max_value=1000)

if st.button("📥 Ingest Uploaded PDFs"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
    else:
        with st.spinner("Processing and ingesting PDFs..."):
            tmp_files = []
            try:
                # Save uploaded files temporarily
                for file in uploaded_files:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                    tmp.write(file.read())
                    tmp.flush()
                    tmp.close()
                    tmp_files.append(tmp.name)

                # Ingest documents into Chroma
                total_chunks = ingest_filepaths(tmp_files, chunk_size, chunk_overlap)
                st.success(f"✅ Successfully ingested {total_chunks} chunks into Chroma DB.")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")
            finally:
                for tmp in tmp_files:
                    try:
                        os.remove(tmp)
                    except:
                        pass

st.markdown("---")

# --- Section 2: Ask Questions ---
st.header("2) Ask Questions & Get Answers")

col1, col2 = st.columns([3, 1])

with col1:
    mode = st.selectbox(
        "Answer mode",
        options=["detailed", "brief", "citations"],
        index=0,
        help="Choose response style: detailed = comprehensive answer; brief = short summary; citations = only sources"
    )
    top_k = st.slider("Number of retrieved documents (Top-K)", min_value=3, max_value=12, value=5)
    question = st.text_input("Ask your question about the ingested PDFs:")

    if st.button("✨ Get Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving and generating answer..."):
                try:
                    result = answer_query(question, top_k=top_k)

                    # Display Answer
                    st.markdown("### ✅ Answer")
                    st.write(result["answer"])

                    # Display Citations
                    citations = result.get("citations", [])
                    if citations:
                        st.markdown("### 📚 Retrieved Evidence")
                        for i, c in enumerate(citations, start=1):
                            st.markdown(f"**{i}. {c['source']} (Page {c['page']})**")
                            st.write(f"> {c['snippet'][:500]}...")
                except Exception as e:
                    st.error(f"Query failed: {e}")

with col2:
    st.markdown("### 📜 Additional Actions")
    if st.button("🧾 Summarize Entire Corpus"):
        with st.spinner("Generating summary of ingested documents..."):
            try:
                retriever = get_retriever(top_k=5)
                docs = retriever.get_relevant_documents(" ")  # fetch docs for summary
                summary = summarize_documents(docs, length="medium")

                st.markdown("### 📝 Summary")
                st.write(summary)

                # Export as PDF
                pdf_bytes = create_summary_pdf(
                    "Arc Corpus Summary",
                    summary,
                    [{"source": d.metadata.get("source", ""), "page": d.metadata.get("page", ""), "snippet": d.page_content[:200]} for d in docs[:6]]
                )

                st.download_button(
                    "⬇️ Download Summary PDF",
                    data=pdf_bytes,
                    file_name="arc_summary.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(f"Summarization failed: {e}")

st.markdown("---")
st.caption("Arc • Academic Research with RAG • Powered by HuggingFace + ChromaDB")
