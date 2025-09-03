import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

def get_pinecone_vectorstore():
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return Pinecone.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
