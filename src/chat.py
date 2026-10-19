import argparse
from ingest import ingest_pdf
from search import search_and_answer

def main():
    parser = argparse.ArgumentParser(description="RAG System CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest PDF into vector database")
    ingest_parser.add_argument("pdf_path", nargs="?", default=None, help="Path to PDF file (optional, uses PDF_PATH from .env)")

    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question about the ingested documents")
    ask_parser.add_argument("question", help="The question to ask")

    args = parser.parse_args()

    if args.command == "ingest":
        if args.pdf_path:
            import os
            os.environ["PDF_PATH"] = args.pdf_path
        try:
            ingest_pdf()
            print("PDF ingested successfully.")
        except Exception as e:
            print(f"Error during ingestion: {e}")

    elif args.command == "ask":
        try:
            answer = search_and_answer(args.question)
            print(f"PERGUNTA: {args.question}")
            print(f"RESPOSTA: {answer}")
        except Exception as e:
            print(f"Error during search: {e}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()