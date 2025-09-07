import os
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as LangPinecone
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME, OPENAI_EMBED_MODEL

pc = Pinecone(api_key=PINECONE_API_KEY)

def create_index_if_not_exists(dimension=1536, metric="cosine"):
    """Create a Pinecone index if it doesn't exist."""
    indexes = [idx["name"] for idx in pc.list_indexes().indexes]
    if PINECONE_INDEX_NAME not in indexes:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"✅ Created Pinecone index: {PINECONE_INDEX_NAME}")
    else:
        print(f"✅ Pinecone index already exists: {PINECONE_INDEX_NAME}")

def get_vector_store():
    """Return a LangChain Pinecone vectorstore."""
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL, openai_api_key=os.getenv("OPENAI_API_KEY"))
    return LangPinecone.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
