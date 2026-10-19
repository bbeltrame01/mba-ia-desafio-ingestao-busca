import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
import psycopg

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@127.0.0.1:5435/rag")
PG_VECTOR_COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "documents")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def validate_vector_extension():
    """
    Verifica se a extensao vector esta criada no PostgreSQL.
    Lanca uma excecao clara se a extensao nao existir.
    """
    try:
        conn = psycopg.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            result = cur.fetchone()
            if not result:
                raise RuntimeError(
                    "Extensao 'vector' nao encontrada no PostgreSQL. "
                    "Execute 'docker-compose down && docker-compose up -d' para recriar os containers."
                )
        conn.close()
    except psycopg.OperationalError as e:
        raise ConnectionError(
            f"Nao conseguiu conectar ao PostgreSQL em {DATABASE_URL}. "
            f"Verifique se os containers estao rodando: {e}"
        )
    except Exception as e:
        raise Exception(f"Erro ao validar extensao vector: {e}")

def ingest_pdf(pdf_path=None):
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set in .env")

    # Validate vector extension before proceeding
    validate_vector_extension()

    # Use provided pdf_path or fall back to env variable
    pdf_file = pdf_path or os.getenv("PDF_PATH", "document.pdf")
    
    # Validate PDF exists
    if not os.path.exists(pdf_file):
        raise FileNotFoundError(f"PDF file not found: {pdf_file}")

    # Load PDF
    loader = PyPDFLoader(pdf_file)
    documents = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    # Generate embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )

    # Store in PGVector
    vector_store = PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        connection=DATABASE_URL,
        collection_name=PG_VECTOR_COLLECTION_NAME,
        use_jsonb=True
    )

    print(f"Ingested {len(chunks)} chunks from {pdf_file}")

if __name__ == "__main__":
    ingest_pdf()