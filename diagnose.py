#!/usr/bin/env python3
"""
Script de diagnóstico para verificar se o sistema RAG está configurado corretamente.
"""

import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_database_connection():
    """Testa conexão com PostgreSQL e presença da extensão vector"""
    try:
        import psycopg
        from dotenv import load_dotenv

        load_dotenv()
        database_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@127.0.0.1:5435/rag")

        print("🔍 Testando conexão com PostgreSQL...")
        conn = psycopg.connect(database_url)
        print("✅ Conexão estabelecida")

        with conn.cursor() as cur:
            # Verificar extensão vector
            cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            if cur.fetchone():
                print("✅ Extensão 'vector' encontrada")
            else:
                print("❌ Extensão 'vector' NÃO encontrada")
                return False

        conn.close()
        return True

    except ImportError:
        print("❌ Módulo psycopg não instalado. Execute: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Erro na conexão: {e}")
        return False

def test_openai_key():
    """Testa se a chave da API do OpenAI está configurada"""
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        print("❌ OPENAI_API_KEY não configurada ou é placeholder")
        return False

    print("✅ OPENAI_API_KEY configurada")
    return True

def test_pdf_file():
    """Testa se o arquivo PDF existe"""
    from dotenv import load_dotenv
    load_dotenv()

    pdf_path = os.getenv("PDF_PATH", "./pdfs/document.pdf")
    if os.path.exists(pdf_path):
        print(f"✅ Arquivo PDF encontrado: {pdf_path}")
        return True
    else:
        print(f"❌ Arquivo PDF não encontrado: {pdf_path}")
        return False

def main():
    print("🚀 DIAGNÓSTICO DO SISTEMA RAG\n")

    tests = [
        ("Conexão PostgreSQL", test_database_connection),
        ("Chave OpenAI", test_openai_key),
        ("Arquivo PDF", test_pdf_file),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        if test_func():
            passed += 1
        print()

    print(f"📊 RESULTADO: {passed}/{total} testes passaram")

    if passed == total:
        print("🎉 Sistema pronto! Você pode executar:")
        print("   python main.py ingest ./pdfs/document.pdf")
        print("   python main.py ask 'Sua pergunta aqui'")
    else:
        print("⚠️  Corrija os problemas acima antes de usar o sistema")

if __name__ == "__main__":
    main()