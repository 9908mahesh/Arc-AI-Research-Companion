from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import os

CHROMA_DIR = "chroma_db"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Save Chroma index (it automatically persists if persist_directory is set)
def save_chroma_index(vectorstore):
    vectorstore.persist()
    print(f"✅ Chroma DB saved at {CHROMA_DIR}")

# ✅ Load Chroma index
def load_chroma_index():
    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(f"❌ Chroma DB not found at {CHROMA_DIR}")
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding_model)
