import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_API_KEY_FREE = os.environ.get("GEMINI_API_KEY_FREE")
# URL은 당장은 안필요함
GEMINI_BASE_URL = os.environ.get("GEMINI_BASE_URL")

CHAT_MODEL = os.environ.get("CHAT_MODEL", "gemini-3.1-pro-preview")
SUMMARY_MODEL = os.environ.get("SUMMARY_MODEL", "gemini-3.1-flash-lite-preview")
MEMORY_MODEL = os.environ.get("MEMORY_MODEL", "gemini-3.1-pro-preview")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "models/gemini-embedding-001")
LOCAL_EMBEDDING_MODEL = os.environ.get("LOCAL_EMBEDDING_MODEL","sentence-transformers/all-MiniLM-L6-v2")