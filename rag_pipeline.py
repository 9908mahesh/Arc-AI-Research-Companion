from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from utils.pinecone_helper import get_pinecone_vectorstore
from config import OPENAI_API_KEY

def get_qa_chain():
    llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=OPENAI_API_KEY)
    vectorstore = get_pinecone_vectorstore()
    return RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
