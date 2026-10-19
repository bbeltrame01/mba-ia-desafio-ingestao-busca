import argparse
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ingest import ingest_pdf
from search import search_and_answer

def main():
    parser = argparse.ArgumentParser(description="RAG System CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest PDF into vector database")
    ingest_parser.add_argument("pdf_path", nargs="?", default=None, help="Path to PDF file or directory (optional, uses PDF_PATH from .env if not provided)")

    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question about the ingested documents")
    ask_parser.add_argument("question", help="The question to ask")

    args = parser.parse_args()

    if args.command == "ingest":
        # Handle both single PDF and directory
        pdf_input = args.pdf_path
        
        if pdf_input is None:
            # No argument provided, use default from .env
            try:
                ingest_pdf()
                print("PDF ingested successfully.")
            except FileNotFoundError as e:
                print(f"Error: {e}")
            except Exception as e:
                print(f"Error during ingestion: {e}")
        elif os.path.isdir(pdf_input):
            # Directory provided - process all PDFs in it
            pdf_files = [f for f in os.listdir(pdf_input) if f.endswith('.pdf')]
            if not pdf_files:
                print(f"No PDF files found in directory: {pdf_input}")
                return
            for pdf_file in pdf_files:
                pdf_path = os.path.join(pdf_input, pdf_file)
                try:
                    ingest_pdf(pdf_path)
                    print(f"✓ {pdf_file} ingested successfully.")
                except Exception as e:
                    print(f"✗ Error during ingestion of {pdf_file}: {e}")
        else:
            # Single PDF file provided
            try:
                ingest_pdf(pdf_input)
                print(f"✓ {os.path.basename(pdf_input)} ingested successfully.")
            except Exception as e:
                print(f"✗ Error during ingestion: {e}")

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