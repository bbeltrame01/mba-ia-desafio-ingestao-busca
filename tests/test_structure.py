"""Estrutura obrigatória do projeto e configuração (docs/Desafio.md)."""
import re

import pytest

ARQUIVOS_OBRIGATORIOS = [
    "docker-compose.yml",
    "requirements.txt",
    ".env.example",
    "src/ingest.py",
    "src/search.py",
    "src/chat.py",
    "README.md",
]


@pytest.mark.parametrize("caminho", ARQUIVOS_OBRIGATORIOS)
def test_arquivo_obrigatorio_existe(root, caminho):
    assert (root / caminho).is_file()


def test_document_pdf_na_raiz(root):
    assert (root / "document.pdf").is_file()


def test_env_example_tem_template_das_variaveis(root):
    env = (root / ".env.example").read_text(encoding="utf-8")
    for var in ("OPENAI_API_KEY", "DATABASE_URL", "PG_VECTOR_COLLECTION_NAME", "PDF_PATH"):
        assert re.search(rf"^{var}=", env, re.M), var


def test_env_example_aponta_para_pdf_existente(root):
    env = (root / ".env.example").read_text(encoding="utf-8")
    pdf_path = re.search(r"^PDF_PATH=(.+)$", env, re.M).group(1).strip().strip("'\"")
    assert (root / pdf_path).is_file()


def test_env_real_nao_versionado(root):
    assert ".env" in (root / ".gitignore").read_text(encoding="utf-8").split()


@pytest.mark.parametrize(
    "pacote",
    ["langchain", "langchain-openai", "langchain-postgres", "langchain-community",
     "langchain-text-splitters", "pypdf", "psycopg", "python-dotenv"],
)
def test_requirements_tem_pacotes_obrigatorios(root, pacote):
    nomes = {
        linha.split("==")[0].strip().lower()
        for linha in (root / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if linha.strip() and not linha.startswith("#")
    }
    assert pacote in nomes


def test_compose_usa_imagem_pgvector(root):
    compose = (root / "docker-compose.yml").read_text(encoding="utf-8")
    assert "pgvector/pgvector" in compose


def test_init_sql_cria_extensao_vector(root):
    sql = (root / "init-db.sql").read_text(encoding="utf-8")
    assert "CREATE EXTENSION IF NOT EXISTS vector" in sql


def test_bootstrap_usa_porta_interna_do_container(root):
    compose = (root / "docker-compose.yml").read_text(encoding="utf-8")
    assert "@postgres:5432/" in compose


def test_readme_explica_execucao(root):
    readme = (root / "README.md").read_text(encoding="utf-8").lower()
    assert "docker compose up" in readme or "docker-compose up" in readme
    assert "ingest" in readme
