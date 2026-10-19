import sys
from search import build_answerer

EXIT_COMMANDS = {"sair", "exit", "quit"}


def run_chat():
    """Chat no terminal: lê perguntas até 'sair' ou EOF (Ctrl+D / Ctrl+Z)."""
    try:
        answer = build_answerer()
    except Exception as e:
        print(f"Erro ao iniciar o chat: {e}")
        return 1

    print("Faça sua pergunta (digite 'sair' para encerrar):")
    while True:
        try:
            question = input("\nPERGUNTA: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            return 0

        try:
            print(f"RESPOSTA: {answer(question)}")
        except Exception as e:
            print(f"Erro ao responder: {e}")


def main():
    return run_chat()


if __name__ == "__main__":
    sys.exit(main())
