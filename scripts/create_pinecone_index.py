import os
import pinecone
from dotenv import load_dotenv

load_dotenv()

# Load environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

# Check if index exists, else create it
if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine"
    )
    print(f"✅ Created Pinecone index: {INDEX_NAME}")
else:
    print(f"✅ Pinecone index already exists: {INDEX_NAME}")
