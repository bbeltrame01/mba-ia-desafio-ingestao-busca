from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres import PGVector
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from db import (
    DATABASE_URL,
    EMBEDDING_MODEL,
    LLM_MODEL,
    OPENAI_API_KEY,
    PG_VECTOR_COLLECTION_NAME,
    validate_vector_extension,
)

TOP_K = 10

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
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY
    )

    # Initialize vector store
    vector_store = PGVector(
        connection=DATABASE_URL,
        collection_name=PG_VECTOR_COLLECTION_NAME,
        embeddings=embeddings,
        use_jsonb=True
    )

    # Vectorize the question and fetch the k most relevant chunks
    def retrieve(question):
        return vector_store.similarity_search_with_score(question, k=TOP_K)

    # Create prompt
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    # Create LLM (gpt-5 models only accept the default temperature)
    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENAI_API_KEY
    )

    # Create chain
    def format_docs(results):
        return "\n\n".join(doc.page_content for doc, _score in results)

    chain = (
        {"contexto": RunnableLambda(retrieve) | format_docs, "pergunta": RunnablePassthrough()}
        | prompt
        | llm
    )

    return chain

def build_answerer():
    """Valida o banco e monta a cadeia uma vez; devolve uma função pergunta -> resposta."""
    validate_vector_extension()
    chain = create_rag_chain()

    def answer(question):
        return chain.invoke(question).content

    return answer

def search_and_answer(question):
    return build_answerer()(question)