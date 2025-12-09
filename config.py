import os
import logging
from dotenv import load_dotenv
load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=LOG_LEVEL,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# OpenAI apenas para router (sem acesso a docs internos)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "gpt-4o-mini")

# Modelos locais (Ollama)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:latest")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")

# Pinecone
# Tenta variável padrão e mantém compatibilidade com nome legado.
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY_DSUNIBLU")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "default")
PINECONE_ENV = os.getenv("PINECONE_ENV", None)  # se estiver usando serverless, pode ignorar
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")

K_RAG = int(os.getenv("RAG_TOP_K", "5"))
