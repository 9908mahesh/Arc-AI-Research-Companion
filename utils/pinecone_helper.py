import os
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangPinecone
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME, OPENAI_EMBED_MODEL

# ✅ Initialize Pinecone client
def init_pinecone():
    if not PINECONE_API_KEY:
        raise RuntimeError("Missing PINECONE_API_KEY")
    return Pinecone(api_key=PINECONE_API_KEY)

# ✅ Create index if it does not exist
def create_index_if_not_exists(dim=1536, metric="cosine"):
    pc = init_pinecone()
    existing_indexes = [i["name"] for i in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=dim,
            metric=metric,
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"✅ Created Pinecone index: {PINECONE_INDEX_NAME}")
        return True
    print(f"ℹ️ Pinecone index already exists: {PINECONE_INDEX_NAME}")
    return False

# ✅ Updated get_vectorstore function
def get_vectorstore():
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL, openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    # ✅ Instead of calling pinecone.Index(), we rely on LangChain wrapper
    # The new LangChain version internally handles the Pinecone client object.
    return LangPinecone.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
