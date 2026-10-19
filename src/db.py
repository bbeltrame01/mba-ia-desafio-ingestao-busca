import os
from dotenv import load_dotenv
import psycopg

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@127.0.0.1:5435/rag")
PG_VECTOR_COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "documents")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-5-nano")


def validate_vector_extension():
    """
    Verifica se a extensão vector está criada no PostgreSQL.
    Lança ConnectionError se o banco estiver inacessível e RuntimeError se a extensão não existir.
    """
    try:
        with psycopg.connect(DATABASE_URL) as conn, conn.cursor() as cur:
            cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            result = cur.fetchone()
    except psycopg.OperationalError as e:
        raise ConnectionError(
            f"Não conseguiu conectar ao PostgreSQL em {DATABASE_URL}. "
            f"Verifique se os containers estão rodando: {e}"
        ) from e

    if not result:
        raise RuntimeError(
            "Extensão 'vector' não encontrada no PostgreSQL. "
            "Execute 'docker compose down && docker compose up -d' para recriar os containers."
        )
