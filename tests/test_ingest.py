"""Requisito 1 do Desafio: ingestão do PDF."""
from unittest.mock import MagicMock

import psycopg
import pytest
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import db
import ingest


@pytest.fixture
def pdf_file(tmp_path):
    path = tmp_path / "document.pdf"
    path.write_bytes(b"%PDF-1.4 fake")
    return path


@pytest.fixture
def pipeline(monkeypatch):
    """Substitui loader, embeddings e PGVector por fakes e devolve os mocks."""
    page = Document(page_content=("Faturamento da empresa SuperTechIABrazil. " * 80), metadata={"page": 0})
    loader = MagicMock()
    loader.return_value.load.return_value = [page]
    splitter = MagicMock(wraps=RecursiveCharacterTextSplitter)
    embeddings = MagicMock()
    pgvector = MagicMock()

    monkeypatch.setattr(ingest, "PyPDFLoader", loader)
    monkeypatch.setattr(ingest, "RecursiveCharacterTextSplitter", splitter)
    monkeypatch.setattr(ingest, "OpenAIEmbeddings", embeddings)
    monkeypatch.setattr(ingest, "PGVector", pgvector)
    return {"loader": loader, "splitter": splitter, "embeddings": embeddings, "pgvector": pgvector}


def test_sem_api_key_falha_antes_de_tudo(monkeypatch):
    monkeypatch.setattr(ingest, "OPENAI_API_KEY", None)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        ingest.ingest_pdf("qualquer.pdf")


def test_pdf_inexistente_gera_file_not_found(fake_key, no_db, tmp_path):
    with pytest.raises(FileNotFoundError):
        ingest.ingest_pdf(str(tmp_path / "nao_existe.pdf"))


def test_usa_pdf_path_do_ambiente_quando_nao_informado(fake_key, no_db, pipeline, pdf_file, monkeypatch):
    monkeypatch.setenv("PDF_PATH", str(pdf_file))
    ingest.ingest_pdf()
    pipeline["loader"].assert_called_once_with(str(pdf_file))


def test_chunks_de_1000_com_overlap_de_150(fake_key, no_db, pipeline, pdf_file):
    ingest.ingest_pdf(str(pdf_file))

    kwargs = pipeline["splitter"].call_args.kwargs
    assert kwargs["chunk_size"] == 1000
    assert kwargs["chunk_overlap"] == 150

    chunks = pipeline["pgvector"].from_documents.call_args.kwargs["documents"]
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 1000 for c in chunks)


def test_embeddings_text_embedding_3_small(fake_key, no_db, pipeline, pdf_file):
    ingest.ingest_pdf(str(pdf_file))
    assert pipeline["embeddings"].call_args.kwargs["model"] == "text-embedding-3-small"


def test_vetores_gravados_no_pgvector(fake_key, no_db, pipeline, pdf_file, monkeypatch):
    monkeypatch.setattr(ingest, "PG_VECTOR_COLLECTION_NAME", "colecao_teste")
    monkeypatch.setattr(ingest, "DATABASE_URL", "postgresql://u:p@host:5435/rag")

    ingest.ingest_pdf(str(pdf_file))

    kwargs = pipeline["pgvector"].from_documents.call_args.kwargs
    assert kwargs["embedding"] is pipeline["embeddings"].return_value
    assert kwargs["collection_name"] == "colecao_teste"
    assert kwargs["connection"] == "postgresql://u:p@host:5435/rag"
    assert all(c.metadata.get("page") == 0 for c in kwargs["documents"])


def test_valida_extensao_antes_de_carregar_pdf(fake_key, pipeline, pdf_file, monkeypatch):
    def boom():
        raise ConnectionError("banco fora")

    monkeypatch.setattr(ingest, "validate_vector_extension", boom)
    with pytest.raises(ConnectionError):
        ingest.ingest_pdf(str(pdf_file))
    pipeline["loader"].assert_not_called()


def test_reingestao_nao_duplica_chunks(fake_key, no_db, pipeline, pdf_file):
    """IDs determinísticos fazem o PGVector sobrescrever (upsert) em vez de duplicar."""
    ingest.ingest_pdf(str(pdf_file))
    ingest.ingest_pdf(str(pdf_file))

    primeira, segunda = (c.kwargs for c in pipeline["pgvector"].from_documents.call_args_list)
    assert primeira["ids"] == segunda["ids"]
    assert len(set(primeira["ids"])) == len(primeira["documents"])


def test_ids_diferentes_para_pdfs_diferentes(fake_key, no_db, pipeline, tmp_path):
    a, b = tmp_path / "a.pdf", tmp_path / "b.pdf"
    for f in (a, b):
        f.write_bytes(b"%PDF")
    ingest.ingest_pdf(str(a))
    ingest.ingest_pdf(str(b))

    ids_a, ids_b = (set(c.kwargs["ids"]) for c in pipeline["pgvector"].from_documents.call_args_list)
    assert ids_a.isdisjoint(ids_b)


def test_pdf_padrao_e_o_document_pdf_da_raiz(fake_key, no_db, pipeline, root, monkeypatch):
    monkeypatch.delenv("PDF_PATH", raising=False)
    ingest.ingest_pdf()
    assert pipeline["loader"].call_args.args[0] == str(root / "document.pdf")


# --- validate_vector_extension -------------------------------------------------

def _fake_connect(row):
    conn = MagicMock()
    conn.__enter__.return_value = conn
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.fetchone.return_value = row
    return MagicMock(return_value=conn), conn


def test_extensao_presente_nao_levanta(monkeypatch):
    connect, conn = _fake_connect(("vector",))
    monkeypatch.setattr(db.psycopg, "connect", connect)
    db.validate_vector_extension()
    conn.__exit__.assert_called_once()


def test_banco_inacessivel_vira_connection_error(monkeypatch):
    monkeypatch.setattr(db.psycopg, "connect", MagicMock(side_effect=psycopg.OperationalError("refused")))
    with pytest.raises(ConnectionError, match="PostgreSQL"):
        db.validate_vector_extension()


def test_extensao_ausente_levanta_runtime_error(monkeypatch):
    connect, _ = _fake_connect(None)
    monkeypatch.setattr(db.psycopg, "connect", connect)
    with pytest.raises(RuntimeError, match="vector"):
        db.validate_vector_extension()


def test_conexao_fechada_mesmo_com_erro(monkeypatch):
    connect, conn = _fake_connect(None)
    monkeypatch.setattr(db.psycopg, "connect", connect)
    with pytest.raises(Exception):
        db.validate_vector_extension()
    conn.__exit__.assert_called_once()
