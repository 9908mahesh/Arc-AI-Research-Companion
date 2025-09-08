from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import CHROMA_DIR
import os

# ✅ HuggingFace Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def save_chroma_index(vectorstore):
    vectorstore.persist()
    print(f"✅ Chroma DB saved at {CHROMA_DIR}")

def load_chroma_index():
    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(f"❌ Chroma DB not found at {CHROMA_DIR}")
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model
    )
