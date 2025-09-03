import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "arc-index")

DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", 5))

def ensure_keys():
    missing = []
    for k,v in [
        ("OPENAI_API_KEY", OPENAI_API_KEY),
        ("PINECONE_API_KEY", PINECONE_API_KEY),
    ]:
        if not v:
            missing.append(k)
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")
