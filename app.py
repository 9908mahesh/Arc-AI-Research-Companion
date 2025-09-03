import streamlit as st
from utils.pdf_loader import load_pdf
from utils.text_splitter import split_text
from utils.pinecone_helper import get_pinecone_vectorstore
from config import OPENAI_API_KEY
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from rag_pipeline import get_qa_chain

import tempfile

st.title("AI Research Companion (Online Mode)")

uploaded_file = st.file_uploader("Upload your research paper (PDF)", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    st.write("Processing document...")
    docs = load_pdf(file_path)
    chunks = split_text(docs)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = get_pinecone_vectorstore()
    vectorstore.add_documents(chunks)

    st.success("Document processed and indexed!")

query = st.text_input("Ask a question about your document:")

if query:
    chain = get_qa_chain()
    response = chain.run(query)
    st.write("### Answer:")
    st.write(response)
