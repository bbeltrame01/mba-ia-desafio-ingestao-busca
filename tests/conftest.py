"""Fixtures compartilhadas.

Nenhum teste acessa rede, OpenAI ou PostgreSQL: clientes externos são
substituídos por fakes. Use xfail para documentar um requisito do Desafio
(docs/Desafio.md) ainda não atendido; com xfail_strict, quando o ajuste for
feito o teste passa a falhar como XPASS e o marcador deve ser removido.
"""
from pathlib import Path

import pytest
from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parent.parent

FALLBACK = "Não tenho informações necessárias para responder sua pergunta."


@pytest.fixture
def root():
    return ROOT


@pytest.fixture
def no_db(monkeypatch):
    """Desliga a checagem da extensão vector nos dois módulos."""
    import ingest
    import search

    monkeypatch.setattr(ingest, "validate_vector_extension", lambda: None)
    monkeypatch.setattr(search, "validate_vector_extension", lambda: None)


@pytest.fixture
def fake_key(monkeypatch):
    import ingest
    import search

    monkeypatch.setattr(ingest, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(search, "OPENAI_API_KEY", "sk-test")


@pytest.fixture
def sample_docs():
    return [
        Document(page_content=f"Trecho {i}: o faturamento foi de 10 milhões de reais.")
        for i in range(10)
    ]
