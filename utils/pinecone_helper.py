import os
import pinecone
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import Pinecone as LangPinecone
from config import PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME, OPENAI_EMBED_MODEL

def init_pinecone():
    if not PINECONE_API_KEY:
        raise RuntimeError("Missing PINECONE_API_KEY")
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

def create_index_if_not_exists(dim=1536, metric="cosine"):
    init_pinecone()
    if PINECONE_INDEX_NAME not in pinecone.list_indexes():
        pinecone.create_index(name=PINECONE_INDEX_NAME, dimension=dim, metric=metric)
        print(f"✅ Created Pinecone index: {PINECONE_INDEX_NAME}")
        return True
    print(f"✅ Pinecone index already exists: {PINECONE_INDEX_NAME}")
    return False

def get_vectorstore():
    init_pinecone()
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL, openai_api_key=os.getenv("OPENAI_API_KEY"))
    return LangPinecone.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
