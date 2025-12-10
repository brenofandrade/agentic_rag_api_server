# Agentic RAG API Server

API em Flask para expor o grafo de recuperação (LangGraph + Pinecone + Ollama/OpenAI).

## Estrutura
- `api_server.py`: ponto de entrada Flask com `create_app`.
- `config.py`: carrega variáveis de ambiente e logging.
- `src/retrieval_graph/`: grafo agentic, configuração e prompts.
- `requirements.txt`: dependências da API.
- `Dockerfile`: build da imagem para produção.

## Variáveis de ambiente (carregadas via `.env`)
- `OPENAI_API_KEY` (router)
- `ROUTER_MODEL` (default: `gpt-4o-mini`)
- `OLLAMA_BASE_URL`, `OLLAMA_CHAT_MODEL`, `OLLAMA_EMBED_MODEL`
- `PINECONE_API_KEY` (ou compatibilidade `PINECONE_API_KEY_DSUNIBLU`), `PINECONE_INDEX_NAME`, `PINECONE_NAMESPACE` (opcional), `PINECONE_ENV` (opcional)
- `RAG_TOP_K`, `PORT`, `LOG_LEVEL`
- (rede corporativa) `REQUESTS_CA_BUNDLE` ou `SSL_CERT_FILE` para CA interna.

## Como rodar localmente
```bash
python -m venv .venv
source .venv/bin/activate  # ou .venv\\Scripts\\activate no Windows
pip install -r requirements.txt
python api_server.py
```

## Docker
Build e execução:
```bash
docker build -t agentic-rag-api .
docker run --env-file .env -p 8000:8000 agentic-rag-api
```

## Endpoints
- `GET /health` — liveness check.
- `POST /chat`
  - Body: `{"question": "...", "history": [{"role": "user"|"assistant", "content": "..."}]}`.
  - Resposta: `answer`, `route` e `router_message` (mensagem amigável do router).
