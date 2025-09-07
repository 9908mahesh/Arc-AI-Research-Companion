import os
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME, OPENAI_EMBED_MODEL

# ✅ Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

def create_index_if_not_exists():
    existing_indexes = [index['name'] for index in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,  # For OpenAI embeddings
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"✅ Created Pinecone index: {PINECONE_INDEX_NAME}")
    else:
        print(f"✅ Pinecone index already exists: {PINECONE_INDEX_NAME}")

def get_vectorstore():
    # ✅ Initialize embeddings
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)

    # ✅ Use LangChain's PineconeVectorStore wrapper
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )

    return vectorstore
