import pinecone
from config import PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

if PINECONE_INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(PINECONE_INDEX_NAME, dimension=1536, metric="cosine")
    print(f"Index {PINECONE_INDEX_NAME} created successfully!")
else:
    print(f"Index {PINECONE_INDEX_NAME} already exists.")
