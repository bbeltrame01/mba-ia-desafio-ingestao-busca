import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres import PGVector
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import psycopg

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@127.0.0.1:5435/rag")
PG_VECTOR_COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "documents")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def validate_vector_extension():
    """
    Verifica se a extensão vector está criada no PostgreSQL.
    Lança uma exceção clara se a extensão não existir.
    """
    try:
        conn = psycopg.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            result = cur.fetchone()
            if not result:
                raise RuntimeError(
                    "Extensao 'vector' nao encontrada no PostgreSQL. "
                    "Execute 'docker-compose down && docker-compose up -d' para recriar os containers."
                )
        conn.close()
    except psycopg.OperationalError as e:
        raise ConnectionError(
            f"Nao conseguiu conectar ao PostgreSQL em {DATABASE_URL}. "
            f"Verifique se os containers estao rodando: {e}"
        )
    except Exception as e:
        raise Exception(f"Erro ao validar extensao vector: {e}")

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

def create_rag_chain():
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set in .env")

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )

    # Initialize vector store
    vector_store = PGVector(
        connection=DATABASE_URL,
        collection_name=PG_VECTOR_COLLECTION_NAME,
        embeddings=embeddings,
        use_jsonb=True
    )

    # Create retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})

    # Create prompt
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    # Create LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        openai_api_key=OPENAI_API_KEY,
        temperature=0
    )

    # Create chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"contexto": retriever | format_docs, "pergunta": RunnablePassthrough()}
        | prompt
        | llm
    )

    return chain

def search_and_answer(question):
    # Validate vector extension before proceeding
    validate_vector_extension()
    
    chain = create_rag_chain()
    response = chain.invoke(question)
    return response.content