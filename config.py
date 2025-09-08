import os
from dotenv import load_dotenv

load_dotenv()

# ✅ OpenAI LLM for answering (still used)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

# ✅ ChromaDB settings
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")

DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", 5))

def ensure_keys():
    missing = []
    for k, v in [
        ("OPENAI_API_KEY", OPENAI_API_KEY),
    ]:
        if not v:
            missing.append(k)
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")
