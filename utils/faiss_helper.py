from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Save FAISS index locally
def save_faiss_index(vectorstore, path="faiss_index"):
    vectorstore.save_local(path)
    print(f"✅ FAISS index saved at {path}")

# ✅ Load FAISS index
def load_faiss_index(path="faiss_index"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ FAISS index not found at {path}")
    return FAISS.load_local(path, embedding_model)
