"""Requisito 2 do Desafio: busca (vetorizar, k=10, prompt, LLM, resposta)."""
import re
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import search
from conftest import FALLBACK


def _prompt_do_desafio(root):
    """Extrai o bloco 'Prompt a ser utilizado' de docs/Desafio.md."""
    text = (root / "docs" / "Desafio.md").read_text(encoding="utf-8")
    bloco = re.search(r"### Prompt a ser utilizado:\s*```\n(.*?)```", text, re.S).group(1)
    return (
        bloco.replace("{resultados concatenados do banco de dados}", "{contexto}")
        .replace("{pergunta do usuário}", "{pergunta}")
        .strip()
    )


def test_prompt_identico_ao_do_desafio(root):
    assert search.PROMPT_TEMPLATE.strip() == _prompt_do_desafio(root)


class FakeStore:
    """Fake do PGVector que aceita retriever ou similarity_search_with_score."""

    def __init__(self, docs, **kwargs):
        self.docs = docs
        self.init_kwargs = kwargs
        self.k = None

    def as_retriever(self, search_kwargs=None, **_):
        self.k = (search_kwargs or {}).get("k")
        return RunnableLambda(lambda _q: self.docs[: self.k])

    def similarity_search_with_score(self, query, k=4, **_):
        self.k = k
        return [(d, 0.1) for d in self.docs[:k]]


@pytest.fixture
def rag(monkeypatch, fake_key, sample_docs):
    """Monta a cadeia com fakes e devolve (store, prompts enviados à LLM, kwargs da LLM)."""
    state = {"prompts": [], "llm_kwargs": None, "store": None, "resposta": "O faturamento foi de 10 milhões de reais."}

    def make_store(**kwargs):
        state["store"] = FakeStore(sample_docs, **kwargs)
        return state["store"]

    def make_llm(**kwargs):
        state["llm_kwargs"] = kwargs

        def call(prompt_value):
            state["prompts"].append(prompt_value.to_string())
            return AIMessage(content=state["resposta"])

        return RunnableLambda(call)

    embeddings = MagicMock()
    monkeypatch.setattr(search, "PGVector", make_store)
    monkeypatch.setattr(search, "ChatOpenAI", make_llm)
    monkeypatch.setattr(search, "OpenAIEmbeddings", embeddings)
    state["embeddings"] = embeddings
    return state


def test_sem_api_key_falha(monkeypatch):
    monkeypatch.setattr(search, "OPENAI_API_KEY", None)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        search.create_rag_chain()


def test_pergunta_vetorizada_com_mesmo_modelo_da_ingestao(rag):
    search.create_rag_chain()
    assert rag["embeddings"].call_args.kwargs["model"] == "text-embedding-3-small"
    assert rag["store"].init_kwargs["embeddings"] is rag["embeddings"].return_value


def test_busca_os_10_resultados_mais_relevantes(rag):
    search.create_rag_chain().invoke("Qual o faturamento?")
    assert rag["store"].k == 10


def test_prompt_enviado_contem_contexto_regras_e_pergunta(rag, sample_docs):
    search.create_rag_chain().invoke("Qual o faturamento da Empresa SuperTechIABrazil?")

    prompt = rag["prompts"][0]
    for doc in sample_docs:
        assert doc.page_content in prompt
    assert "Responda somente com base no CONTEXTO." in prompt
    assert "PERGUNTA DO USUÁRIO:\nQual o faturamento da Empresa SuperTechIABrazil?" in prompt


def test_llm_nao_forca_temperatura(rag):
    # Modelos gpt-5 só aceitam a temperatura padrão
    search.create_rag_chain()
    assert "temperature" not in rag["llm_kwargs"]


def test_llm_gpt_5_nano(rag):
    search.create_rag_chain()
    assert rag["llm_kwargs"]["model"] == "gpt-5-nano"


def test_search_and_answer_retorna_texto_da_llm(rag, no_db):
    assert search.search_and_answer("Qual o faturamento?") == "O faturamento foi de 10 milhões de reais."


def test_pergunta_fora_do_contexto_devolve_mensagem_padrao(rag, no_db):
    rag["resposta"] = FALLBACK
    assert search.search_and_answer("Quantos clientes temos em 2024?") == FALLBACK


def test_build_answerer_monta_cadeia_uma_vez(monkeypatch, fake_key, no_db):
    chain = MagicMock()
    chain.invoke.side_effect = lambda q: AIMessage(content=f"r:{q}")
    create = MagicMock(return_value=chain)
    monkeypatch.setattr(search, "create_rag_chain", create)

    answer = search.build_answerer()
    assert [answer("a"), answer("b")] == ["r:a", "r:b"]
    create.assert_called_once()


def test_search_and_answer_valida_extensao(monkeypatch, fake_key):
    def boom():
        raise ConnectionError("banco fora")

    chain = MagicMock()
    monkeypatch.setattr(search, "validate_vector_extension", boom)
    monkeypatch.setattr(search, "create_rag_chain", chain)
    with pytest.raises(ConnectionError):
        search.search_and_answer("x")
    chain.assert_not_called()
