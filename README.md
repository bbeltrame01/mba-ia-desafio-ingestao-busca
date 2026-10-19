# Sistema de Ingestão e Busca Semântica com LangChain e PostgreSQL

Este projeto implementa um sistema de Retrieval-Augmented Generation (RAG) para ingestão e busca semântica em documentos PDF, utilizando LangChain, PostgreSQL com pgVector e OpenAI.

## Pré-requisitos

- Python 3.12
- Docker e Docker Compose
- Chave da API do OpenAI

## Instalação

1. Clone o repositório:
   ```bash
   git clone <url-do-repositorio>
   cd mba-ia-desafio-ingestao-busca
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure as variáveis de ambiente:
   ```bash
   cp .env.example .env
   ```
   Edite o arquivo `.env` e adicione sua chave da API do OpenAI:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5435/rag
   PG_VECTOR_COLLECTION_NAME=documents
   PDF_PATH=./pdfs/document.pdf
   ```

4. Inicie o banco de dados PostgreSQL com pgVector:
   ```bash
   docker-compose up -d
   ```

5. Aguarde o banco estar pronto (cerca de 30 segundos). A extensão pgVector será criada automaticamente.

## Uso

### Ingestão de PDF

Para ingerir um PDF no sistema:

```bash
python main.py ingest ./pdfs/
```

Substitua `./pdfs/` pelo diretório contendo os arquivos PDF. O sistema processará todos os PDFs no diretório, dividindo-os em chunks, gerando embeddings e armazenando no banco vetorial.

### Consulta via CLI

Para fazer uma pergunta sobre o conteúdo dos PDFs ingeridos:

```bash
python main.py ask "Qual o faturamento da Empresa SuperTechIABrazil?"
```

O sistema retornará uma resposta baseada exclusivamente no conteúdo dos documentos.

### Exemplo de saída:

```
PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.
```

Para perguntas fora do contexto:

```
PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

## Arquitetura

O sistema segue o fluxo RAG clássico:

1. **PDFs** → Loader (PyPDFLoader)
2. **Texto** → Chunking (RecursiveCharacterTextSplitter, chunk_size=1000, overlap=150)
3. **Chunks** → Embeddings (OpenAI text-embedding-3-small)
4. **Vetores** → PostgreSQL + pgVector (porta 5435)
5. **Busca** → Similarity Search (top-k=10)
6. **Contexto** → LLM (GPT-3.5-turbo) com prompt específico
7. **Resposta** → CLI

## Melhorias Implementadas

- **Validação de Extensão**: Verificação automática da extensão pgVector antes das operações
- **Tratamento de Erros**: Mensagens claras para problemas de configuração
- **Flexibilidade de PDF**: Suporte a arquivo único ou diretório de PDFs
- **Porta Configurável**: PostgreSQL exposto na porta 5435 para evitar conflitos
- **Inicialização Automática**: Banco e extensão criados automaticamente no startup

## Estrutura do Projeto

```
├── docker-compose.yml          # Configuração do PostgreSQL com pgVector
├── requirements.txt            # Dependências Python
├── .env.example                # Template das variáveis de ambiente
├── main.py                     # CLI principal
├── document.pdf                # PDF de exemplo para ingestão
├── docs/                       # Documentação do desenvolvimento
├── src/
│   ├── ingest.py               # Script de ingestão do PDF
│   ├── search.py               # Lógica de busca e RAG
│   └── chat.py                 # CLI alternativo (não usado no main)
└── README.md                   # Este arquivo
```

## Tecnologias Utilizadas

- **Python 3.12**
- **LangChain**: Framework para aplicações LLM
- **PostgreSQL + pgVector**: Banco vetorial
- **OpenAI**: Embeddings e LLM
- **Docker**: Containerização do banco

## Desenvolvimento

O sistema foi desenvolvido seguindo boas práticas de engenharia de software:

- Separação de responsabilidades (SRP)
- Configuração via variáveis de ambiente
- Tratamento de erros
- Comentários explicativos
- Estrutura escalável

## Troubleshooting

- **Erro de conexão com o banco**: Verifique se o Docker Compose está rodando (`docker-compose ps`)
- **Chave da API inválida**: Verifique se `OPENAI_API_KEY` está correta no `.env`
- **PDF não encontrado**: Certifique-se de que o arquivo PDF existe no caminho especificado
- **Rate limit da OpenAI**: Aguarde alguns minutos e tente novamente

## Limitações

- Suporta apenas PDFs
- Respostas baseadas exclusivamente no contexto dos documentos ingeridos
- Não mantém histórico de conversas