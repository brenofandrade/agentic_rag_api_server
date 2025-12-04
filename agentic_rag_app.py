import os
import logging
import uuid
from typing import List, Literal, Optional, Dict, Any

from config import (
    OPENAI_API_KEY,
    ROUTER_MODEL,
    OLLAMA_CHAT_MODEL,
    OLLAMA_EMBED_MODEL,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_ENV,
    K_RAG,
    logger,
)   

from flask import Flask, request, jsonify
from flask_cors import CORS

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI

from langchain_core.documents import Document
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage,
)
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated

from agents import AgentRouter, AgentState


# -------------------------------------------------------------------
# 1. LLMs e Vector Store
# -------------------------------------------------------------------

# Router LLM (OpenAI) - não expõe documentos
router_llm = ChatOpenAI(
    model=ROUTER_MODEL,
    api_key=OPENAI_API_KEY,
    temperature=0.0,
)

# LLM local para respostas (com ou sem RAG)
local_llm = ChatOllama(
    model=OLLAMA_CHAT_MODEL,
    temperature=0.1,
)

# Embeddings locais
embeddings = OllamaEmbeddings(
    model=OLLAMA_EMBED_MODEL
)

# Pinecone client e vector store
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
index = pinecone_client.Index(PINECONE_INDEX_NAME)

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text",
)

# Retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": K_RAG}
)




# -------------------------------
# Funções de nós do Graph Agentic  
# -------------------------------

with open("prompts/router_system_prompt.txt", "r", encoding="utf-8") as f:
    ROUTER_SYSTEM_PROMPT = f.read()

with open("prompts/rag_system_prompt.txt", "r", encoding="utf-8") as f:
    RAG_SYSTEM_PROMPT = f.read()

with open("prompts/direct_system_prompt.txt", "r", encoding="utf-8") as f:
    DIRECT_SYSTEM_PROMPT = f.read()



def route_question(state: AgentState) -> AgentState:
    """Roteia a pergunta usando o LLM da OpenAI (sem acesso aos documentos)."""
    question = state["question"]
    logger.info(f"[ROUTER] Classificando pergunta: {question}")

    messages = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]

    response = router_llm.invoke(messages)

    import json
    try:
        router_output = json.loads(response.content)
        label = router_output.get("label")
        reason = router_output.get("reason", "")
        message_to_user = router_output.get("message_to_user", "")
    except Exception as e:
        logger.exception("Erro ao parsear saída do router. Caindo em 'general_knowledge'.")
        label = "general_knowledge"
        reason = f"Falha ao parsear JSON do router: {e}"
        message_to_user = "Vou responder com meu conhecimento geral."

    state["route"] = AgentRouter(
        label=label,  # type: ignore
        reason=reason,
        message_to_user=message_to_user,
    )

    # adiciona a mensagem amigável como primeira resposta no histórico
    state["messages"].append(AIMessage(content=message_to_user))

    return state


def retrieve_docs(state: AgentState) -> AgentState:
    """Faz retrieve de documentos no Pinecone."""
    question = state["question"]
    logger.info(f"[RAG] Recuperando documentos para: {question}")

    docs = retriever.invoke(question)  # retorna List[Document]
    if not isinstance(docs, list):
        docs = [docs]

    logger.info(f"[RAG] {len(docs)} documentos recuperados.")
    state["docs"] = docs
    return state


def answer_with_rag(state: AgentState) -> AgentState:
    """Gera resposta usando contexto dos documentos (LLM local)."""
    docs = state.get("docs", [])
    question = state["question"]

    # Constrói contexto concatenando os docs
    context_parts = []
    for i, doc in enumerate(docs):
        meta_str = ", ".join([f"{k}: {v}" for k, v in (doc.metadata or {}).items()])
        context_parts.append(f"[DOC {i+1} | {meta_str}]\n{doc.page_content}")

    context = "\n\n".join(context_parts) if context_parts else "Nenhum documento relevante foi encontrado."

    messages = [
        SystemMessage(content=RAG_SYSTEM_PROMPT),
        HumanMessage(content=f"Pergunta: {question}\n\nContexto:\n{context}"),
    ]

    logger.info("[RAG] Gerando resposta com contexto...")
    response = local_llm.invoke(messages)

    answer = response.content
    state["answer"] = answer
    state["messages"].append(AIMessage(content=answer))

    return state


def answer_direct(state: AgentState) -> AgentState:
    """Responde usando apenas conhecimento geral (LLM local)."""
    question = state["question"]
    messages = [
        SystemMessage(content=DIRECT_SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]

    logger.info("[DIRECT] Respondendo com conhecimento geral...")
    response = local_llm.invoke(messages)
    answer = response.content
    state["answer"] = answer
    state["messages"].append(AIMessage(content=answer))
    return state


def handle_clarify(state: AgentState) -> AgentState:
    """Gera pergunta de esclarecimento para o usuário."""
    question = state["question"]
    prompt = (
        "O usuário fez a seguinte pergunta, que está ambígua ou incompleta:\n"
        f"\"{question}\"\n\n"
        "Formule UMA pergunta clara e direta para pedir mais detalhes, em português."
    )
    response = local_llm.invoke([HumanMessage(content=prompt)])
    answer = response.content
    state["answer"] = answer
    state["messages"].append(AIMessage(content=answer))
    return state


def handle_greeting(state: AgentState) -> AgentState:
    """Responde a saudações de forma amigável."""
    question = state["question"]
    prompt = (
        "O usuário enviou uma saudação ou conversa informal:\n"
        f"\"{question}\"\n\n"
        "Responda de forma simpática, breve, em português, e ofereça ajuda."
    )
    response = local_llm.invoke([HumanMessage(content=prompt)])
    answer = response.content
    state["answer"] = answer
    state["messages"].append(AIMessage(content=answer))
    return state


def handle_farewell(state: AgentState) -> AgentState:
    """Responde despedidas de forma amigável."""
    question = state["question"]
    prompt = (
        "O usuário está encerrando a conversa:\n"
        f"\"{question}\"\n\n"
        "Responda de forma educada, agradecendo e se colocando à disposição no futuro, em português."
    )
    response = local_llm.invoke([HumanMessage(content=prompt)])
    answer = response.content
    state["answer"] = answer
    state["messages"].append(AIMessage(content=answer))
    return state


# -------------------------------
# Construção do Graph Agentic RAG 
# ------------------------------- 

graph_builder = StateGraph(AgentState)

graph_builder.add_node("router", route_question)
graph_builder.add_node("retrieve_docs", retrieve_docs)
graph_builder.add_node("answer_with_rag", answer_with_rag)
graph_builder.add_node("answer_direct", answer_direct)
graph_builder.add_node("clarify", handle_clarify)
graph_builder.add_node("greeting", handle_greeting)
graph_builder.add_node("farewell", handle_farewell)

graph_builder.add_edge(START, "router")


def route_from_state(state: AgentState) -> str:
    """Decide próximo nó com base na rota definida pelo router."""
    route = state.get("route")
    if not route:
        logger.warning("Route ausente no estado. Usando 'general_knowledge'.")
        return "answer_direct"

    label = route["label"]
    if label == "internal_docs_with_rag":
        return "retrieve_docs"
    elif label == "general_knowledge":
        return "answer_direct"
    elif label == "clarify":
        return "clarify"
    elif label == "greeting":
        return "greeting"
    elif label == "farewell":
        return "farewell"
    else:
        logger.warning(f"Label desconhecido '{label}'. Usando 'answer_direct'.")
        return "answer_direct"


# Router condicional a partir do nó "router"
graph_builder.add_conditional_edges(
    "router",
    route_from_state,
    {
        "retrieve_docs": "retrieve_docs",
        "answer_direct": "answer_direct",
        "clarify": "clarify",
        "greeting": "greeting",
        "farewell": "farewell",
    },
)

# Depois de recuperar docs, vai para geração de resposta com RAG e termina
graph_builder.add_edge("retrieve_docs", "answer_with_rag")
graph_builder.add_edge("answer_with_rag", END)
graph_builder.add_edge("answer_direct", END)
graph_builder.add_edge("clarify", END)
graph_builder.add_edge("greeting", END)
graph_builder.add_edge("farewell", END)

agent_graph = graph_builder.compile()


# ------------------ Flask App API Server ------------------

app = Flask(__name__)
CORS(app)

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True) or {}
    question:str = data.get("question").strip()
    session_id: str = data.get("session_id")

    if not question:
        return jsonify({"error":"Campo 'question' é obrigatório."}), 400

    messages: List[BaseMessage] = []

    # for msg in history:
    #     role = msg.get("role")
    #     content = msg.get("content")
    #     history: List[Dict[str, Any]] = data.get("history", [])

    #     if role == "user":
    #         messages.append(HumanMessage(content=content))
    #     elif role == "assistant":
    #         messages.append(AIMessage(content=content))

    # Inicializa estado do agente
    initial_state: AgentState = {
        "session_id": session_id,
        "question": question,
        "messages": messages,
        "docs": [],
        "route": None,
        "answer": None,
    }

    logger.info(f"[CHAT] Nova requisição  | Session ID: {session_id} | Pergunta: {question}")

    final_state = agent_graph.invoke(initial_state)

    answer = final_state.get("answer")
    route = final_state.get("route")

    logger.info(f"[CHAT] | Pergunta: {question} | Resposta: {answer} | Rota: {route}")

    return jsonify(
        {
            "answer": answer,
            "route": route,
        }
    )


def test_chat_endpoint():
    """Função simples para testar a rota /chat localmente."""
    import requests

    url = os.getenv("CHAT_ENDPOINT_URL", "http://localhost:5000/chat")
    payload = {
        "question": "O que é sinistralidade?",
        "session_id": "teste-session-123",
        "history": [],
    }

    print(f"Enviando requisição de teste para {url}...")
    resp = requests.post(url, json=payload)
    print("Status:", resp.status_code)
    try:
        print("Resposta JSON:", resp.json())
    except Exception:
        print("Resposta bruta:", resp.text)


if __name__ == "__main__":

    test_chat_endpoint() # Descomente para testar a API localmente

    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=True)