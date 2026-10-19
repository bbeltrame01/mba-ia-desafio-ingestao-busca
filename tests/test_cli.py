"""Consulta via CLI: chat.py (exigido pelo Desafio) e main.py (proposto no Esboço)."""
import sys
from unittest.mock import MagicMock

import pytest

import chat
import main
from conftest import FALLBACK


def _run(module, monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", [module.__name__ + ".py", *args])
    return module.main()


# --- main.py -------------------------------------------------------------------

def test_main_ask_imprime_pergunta_e_resposta(monkeypatch, capsys):
    monkeypatch.setattr(main, "search_and_answer", lambda q: "O faturamento foi de 10 milhões de reais.")
    code = _run(main, monkeypatch, "ask", "Qual o faturamento da Empresa SuperTechIABrazil?")

    out = capsys.readouterr().out
    assert code == 0
    assert "PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?" in out
    assert "RESPOSTA: O faturamento foi de 10 milhões de reais." in out


def test_main_ingest_sem_argumento_usa_padrao(monkeypatch):
    ingest_pdf = MagicMock()
    monkeypatch.setattr(main, "ingest_pdf", ingest_pdf)
    assert _run(main, monkeypatch, "ingest") == 0
    ingest_pdf.assert_called_once_with()


def test_main_ingest_arquivo(monkeypatch, tmp_path):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF")
    ingest_pdf = MagicMock()
    monkeypatch.setattr(main, "ingest_pdf", ingest_pdf)
    assert _run(main, monkeypatch, "ingest", str(pdf)) == 0
    ingest_pdf.assert_called_once_with(str(pdf))


def test_main_ingest_diretorio_processa_cada_pdf(monkeypatch, tmp_path):
    for name in ("a.pdf", "b.pdf", "notas.txt"):
        (tmp_path / name).write_bytes(b"x")
    ingest_pdf = MagicMock()
    monkeypatch.setattr(main, "ingest_pdf", ingest_pdf)

    _run(main, monkeypatch, "ingest", str(tmp_path))

    chamados = sorted(c.args[0] for c in ingest_pdf.call_args_list)
    assert chamados == sorted(str(tmp_path / n) for n in ("a.pdf", "b.pdf"))


def test_main_ingest_diretorio_vazio_avisa(monkeypatch, tmp_path, capsys):
    ingest_pdf = MagicMock()
    monkeypatch.setattr(main, "ingest_pdf", ingest_pdf)
    assert _run(main, monkeypatch, "ingest", str(tmp_path)) == 1
    ingest_pdf.assert_not_called()
    assert "No PDF files found" in capsys.readouterr().out


def test_main_ingest_diretorio_aceita_extensao_maiuscula(monkeypatch, tmp_path):
    (tmp_path / "RELATORIO.PDF").write_bytes(b"x")
    ingest_pdf = MagicMock()
    monkeypatch.setattr(main, "ingest_pdf", ingest_pdf)
    _run(main, monkeypatch, "ingest", str(tmp_path))
    ingest_pdf.assert_called_once()


def test_main_ask_com_falha_retorna_codigo_1(monkeypatch):
    monkeypatch.setattr(main, "search_and_answer", MagicMock(side_effect=ConnectionError("banco fora")))
    assert _run(main, monkeypatch, "ask", "x") == 1


def test_main_ingest_com_falha_parcial_retorna_codigo_1(monkeypatch, tmp_path):
    for name in ("a.pdf", "b.pdf"):
        (tmp_path / name).write_bytes(b"x")
    ingest_pdf = MagicMock(side_effect=[None, ValueError("corrompido")])
    monkeypatch.setattr(main, "ingest_pdf", ingest_pdf)
    assert _run(main, monkeypatch, "ingest", str(tmp_path)) == 1
    assert ingest_pdf.call_count == 2


def test_main_chat_abre_o_chat_interativo(monkeypatch):
    run_chat = MagicMock(return_value=0)
    monkeypatch.setattr(main, "run_chat", run_chat)
    assert _run(main, monkeypatch, "chat") == 0
    run_chat.assert_called_once_with()


# --- chat.py -------------------------------------------------------------------

def _entradas(monkeypatch, *linhas):
    """Simula o usuário digitando; ao acabar as linhas, envia EOF (Ctrl+D/Ctrl+Z)."""
    fila = list(linhas)

    def fake_input(prompt=""):
        print(prompt, end="")
        if not fila:
            raise EOFError
        return fila.pop(0)

    monkeypatch.setattr("builtins.input", fake_input)


@pytest.fixture
def respostas(monkeypatch):
    """Substitui a cadeia RAG por respostas fixas e conta quantas vezes foi montada."""
    tabela = {
        "Qual o faturamento da Empresa SuperTechIABrazil?": "O faturamento foi de 10 milhões de reais.",
        "Quantos clientes temos em 2024?": FALLBACK,
    }
    build = MagicMock(return_value=lambda q: tabela[q])
    monkeypatch.setattr(chat, "build_answerer", build)
    return tabela, build


def test_chat_sem_argumentos_abre_chat_interativo(monkeypatch, capsys, respostas):
    tabela, _ = respostas
    _entradas(monkeypatch, *tabela)

    assert _run(chat, monkeypatch) == 0

    out = capsys.readouterr().out
    assert "Faça sua pergunta" in out
    assert "RESPOSTA: O faturamento foi de 10 milhões de reais." in out
    assert f"RESPOSTA: {FALLBACK}" in out


def test_chat_monta_a_cadeia_uma_vez_por_sessao(monkeypatch, respostas):
    tabela, build = respostas
    _entradas(monkeypatch, *tabela)
    chat.run_chat()
    build.assert_called_once()


def test_chat_encerra_com_sair_e_ignora_linha_vazia(monkeypatch, capsys, respostas):
    _entradas(monkeypatch, "", "  ", "sair", "Quantos clientes temos em 2024?")
    assert chat.run_chat() == 0
    assert "RESPOSTA" not in capsys.readouterr().out


def test_chat_continua_apos_erro_em_uma_pergunta(monkeypatch, capsys):
    def answer(q):
        if q == "quebra":
            raise TimeoutError("OpenAI lenta")
        return "ok"

    monkeypatch.setattr(chat, "build_answerer", lambda: answer)
    _entradas(monkeypatch, "quebra", "outra")

    assert chat.run_chat() == 0
    out = capsys.readouterr().out
    assert "Erro ao responder: OpenAI lenta" in out
    assert "RESPOSTA: ok" in out


def test_chat_falha_ao_iniciar_retorna_codigo_1(monkeypatch, capsys):
    monkeypatch.setattr(chat, "build_answerer", MagicMock(side_effect=RuntimeError("Extensão 'vector' não encontrada")))
    assert chat.run_chat() == 1
    assert "Erro ao iniciar o chat" in capsys.readouterr().out
