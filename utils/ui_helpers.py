import streamlit as st

def style_app():
    st.markdown(
        """
        <style>
        .reportview-container .main .block-container{padding:1rem 3rem;}
        .stButton>button {background-color:#0b5fff; color: white;}
        </style>
        """, unsafe_allow_html=True
    )

def sidebar_instructions():
    st.sidebar.header("How to use Arc")
    st.sidebar.markdown(
        "- Upload one or more PDFs.\n"
        "- Click **Ingest** to index documents (Pinecone).\n"
        "- Ask a question. Choose answer mode.\n"
        "- Use **Summarize** to produce an exportable summary.\n"
    )
