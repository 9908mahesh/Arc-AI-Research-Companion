import pinecone
from langchain.embeddings import OpenAIEmbeddings
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
        return True
    return False

def get_vectorstore():
    init_pinecone()
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL, openai_api_key=None)  # langchain will read env OPENAI_API_KEY
    return LangPinecone(index_name=PINECONE_INDEX_NAME, embedding=embeddings, pinecone_index=pinecone.Index(PINECONE_INDEX_NAME))
