import os
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

from db import (
    DATABASE_URL,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    PG_VECTOR_COLLECTION_NAME,
    validate_vector_extension,
)

DEFAULT_PDF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "document.pdf")


def chunk_ids(pdf_file, chunks):
    """
    IDs determinísticos por arquivo e posição do chunk: reingerir o mesmo PDF
    sobrescreve os registros (upsert) em vez de duplicá-los.
    """
    source = os.path.abspath(pdf_file)
    return [
        str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}#{chunk.metadata.get('page')}#{i}"))
        for i, chunk in enumerate(chunks)
    ]


def ingest_pdf(pdf_path=None):
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set in .env")

    # Validate vector extension before proceeding
    validate_vector_extension()

    # Use provided pdf_path or fall back to env variable
    pdf_file = pdf_path or os.getenv("PDF_PATH") or DEFAULT_PDF

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
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY
    )

    # Store in PGVector
    PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        ids=chunk_ids(pdf_file, chunks),
        connection=DATABASE_URL,
        collection_name=PG_VECTOR_COLLECTION_NAME,
        use_jsonb=True
    )

    print(f"Ingested {len(chunks)} chunks from {pdf_file}")

if __name__ == "__main__":
    ingest_pdf()
