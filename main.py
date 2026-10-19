import argparse
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ingest import ingest_pdf
from search import search_and_answer
from chat import run_chat


def ingest_command(pdf_input):
    """Ingere um PDF, todos os PDFs de um diretório ou o PDF_PATH do .env. Retorna o exit code."""
    if pdf_input is None:
        pdf_paths = [None]
    elif os.path.isdir(pdf_input):
        pdf_paths = [
            os.path.join(pdf_input, f)
            for f in sorted(os.listdir(pdf_input))
            if f.lower().endswith('.pdf')
        ]
        if not pdf_paths:
            print(f"No PDF files found in directory: {pdf_input}")
            return 1
    else:
        pdf_paths = [pdf_input]

    failures = 0
    for pdf_path in pdf_paths:
        name = os.path.basename(pdf_path) if pdf_path else "PDF_PATH"
        try:
            ingest_pdf(pdf_path) if pdf_path else ingest_pdf()
            print(f"✓ {name} ingested successfully.")
        except Exception as e:
            failures += 1
            print(f"✗ Error during ingestion of {name}: {e}")
    return 1 if failures else 0


def ask_command(question):
    try:
        answer = search_and_answer(question)
    except Exception as e:
        print(f"Error during search: {e}")
        return 1
    print(f"PERGUNTA: {question}")
    print(f"RESPOSTA: {answer}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="RAG System CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest PDF into vector database")
    ingest_parser.add_argument("pdf_path", nargs="?", default=None, help="Path to PDF file or directory (optional, uses PDF_PATH from .env if not provided)")

    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question about the ingested documents")
    ask_parser.add_argument("question", help="The question to ask")

    # Chat command
    subparsers.add_parser("chat", help="Interactive chat in the terminal")

    args = parser.parse_args()

    if args.command == "ingest":
        return ingest_command(args.pdf_path)
    if args.command == "ask":
        return ask_command(args.question)
    if args.command == "chat":
        return run_chat()

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
