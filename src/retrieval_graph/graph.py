from functools import partial
from typing import List, Literal, TypedDict, Optional

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import StateGraph, START, END
from pinecone import Pinecone

from .configuration import AgentConfiguration





class Router(TypedDict):
    label: Literal["internal_docs_with_rag", "general_knowledge", "clarify", "greeting", "farewell"]
    reason: str
    message_to_user: str


class AgentState(TypedDict):
    question: str
    messages: List[BaseMessage]
    docs: List[Document]
    route: Optional[Router]
    answer: Optional[str]
    retrieval_error: Optional[str]


def _build_services(configuration: AgentConfiguration):
    router_llm = ChatOpenAI(
        model=configuration.router_model,
        api_key=configuration.openai_api_key,
        temperature=0.0,
    )
    generator_llm = ChatOllama(
        model=configuration.generator_model,
        base_url=configuration.ollama_base_url,
        temperature=0.1,
    )
    embeddings = OllamaEmbeddings(
        model=configuration.embed_model,
        base_url=configuration.ollama_base_url,
    )

    if not configuration.pinecone_api_key:
        raise ValueError("PINECONE_API_KEY is required for retrieval.")
    pc_kwargs = {"api_key": configuration.pinecone_api_key}
    if configuration.pinecone_env:
        pc_kwargs["environment"] = configuration.pinecone_env
    pc = Pinecone(**pc_kwargs)
    index = pc.Index(configuration.pinecone_index_name)
    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=configuration.pinecone_namespace,
    )
    retriever = vector_store.as_retriever(
        search_kwargs={"k": configuration.rag_top_k, "namespace": configuration.pinecone_namespace}
    )
    return router_llm, generator_llm, retriever


def route_query(state: AgentState, *, config: RunnableConfig, configuration: AgentConfiguration, router_llm: ChatOpenAI) -> AgentState:
    messages = [
        SystemMessage(content=configuration.router_system_prompt),
        HumanMessage(content=state["question"]),
    ]
    response = router_llm.invoke(messages)
    import json

    try:
        payload = json.loads(response.content)
        label = payload.get("label", "general_knowledge")
        reason = payload.get("reason", "")
        message_to_user = payload.get("message_to_user", "")
    except Exception as exc:
        label = "general_knowledge"
        reason = f"Falha ao parsear JSON do router: {exc}"
        message_to_user = "Vou responder com meu conhecimento geral."

    state["route"] = Router(label=label, reason=reason, message_to_user=message_to_user)  # type: ignore[arg-type]
    state["messages"].append(AIMessage(content=message_to_user))
    return state


def retrieve_docs(state: AgentState, *, retriever) -> AgentState:
    try:
        docs = retriever.invoke(state["question"])
        docs_list = docs if isinstance(docs, list) else [docs]
        if not docs_list:
            state["retrieval_error"] = "Não consegui acessar os documentos no momento."
        state["docs"] = docs_list
    except Exception as exc:
        state["retrieval_error"] = "Não consegui acessar os documentos no momento."
        state["docs"] = []
        return state
    return state


def answer_with_rag(state: AgentState, *, configuration: AgentConfiguration, generator_llm: ChatOllama) -> AgentState:
    retrieval_error = state.get("retrieval_error")
    docs = state.get("docs", [])

    if retrieval_error or not docs:
        answer = retrieval_error or "Não consegui acessar os documentos no momento."
        state["answer"] = answer
        state["messages"].append(AIMessage(content=answer))
        return state

    context_parts = []
    for i, doc in enumerate(docs):
        meta_str = ", ".join([f"{k}: {v}" for k, v in (doc.metadata or {}).items()])
        context_parts.append(f"[DOC {i+1} | {meta_str}]\n{doc.page_content}")
    context = "\n\n".join(context_parts)

    messages = [
        SystemMessage(content=configuration.response_system_prompt),
        HumanMessage(content=f"Pergunta: {state['question']}\n\nContexto:\n{context}"),
    ]
    response = generator_llm.invoke(messages)
    answer = response.content
    state["answer"] = answer
    state["messages"].append(AIMessage(content=answer))
    return state


def answer_direct(state: AgentState, *, configuration: AgentConfiguration, generator_llm: ChatOllama) -> AgentState:
    messages = [
        SystemMessage(content=configuration.general_system_prompt),
        HumanMessage(content=state["question"]),
    ]
    response = generator_llm.invoke(messages)
    answer = response.content
    state["answer"] = answer
    state["messages"].append(AIMessage(content=answer))
    return state


def handle_clarify(state: AgentState, *, configuration: AgentConfiguration, generator_llm: ChatOllama) -> AgentState:
    messages = [
        SystemMessage(content=configuration.more_info_system_prompt),
        HumanMessage(content=state["question"]),
    ]
    response = generator_llm.invoke(messages)
    answer = response.content
    state["answer"] = answer
    state["messages"].append(AIMessage(content=answer))
    return state


def handle_greeting(state: AgentState, *, generator_llm: ChatOllama) -> AgentState:
    response = generator_llm.invoke([HumanMessage(content="Responda de forma simpática e breve: " + state["question"])])
    answer = response.content
    state["answer"] = answer
    state["messages"].append(AIMessage(content=answer))
    return state


def handle_farewell(state: AgentState, *, generator_llm: ChatOllama) -> AgentState:
    prompt = (
        "O usuário está encerrando a conversa.\n"
        f'Entrada: "{state["question"]}"\n\n'
        "Responda apenas com uma despedida cordial e direta em português (1 frase curta). "
        "Não explique como responderia; apenas dê a despedida."
    )
    response = generator_llm.invoke([HumanMessage(content=prompt)])
    answer = response.content
    state["answer"] = answer
    state["messages"].append(AIMessage(content=answer))
    return state


def _route_from_state(state: AgentState) -> str:
    route = state.get("route")
    label = route["label"] if isinstance(route, dict) else None
    return {
        "internal_docs_with_rag": "retrieve_docs",
        "general_knowledge": "answer_direct",
        "clarify": "clarify",
        "greeting": "greeting",
        "farewell": "farewell",
    }.get(label, "answer_direct")


def build_graph(config: RunnableConfig | None = None):
    configuration = AgentConfiguration.from_runnable_config(config or {})
    router_llm, generator_llm, retriever = _build_services(configuration)

    graph = StateGraph(AgentState)
    graph.add_node("router", partial(route_query, configuration=configuration, router_llm=router_llm))
    graph.add_node("retrieve_docs", partial(retrieve_docs, retriever=retriever))
    graph.add_node("answer_with_rag", partial(answer_with_rag, configuration=configuration, generator_llm=generator_llm))
    graph.add_node("answer_direct", partial(answer_direct, configuration=configuration, generator_llm=generator_llm))
    graph.add_node("clarify", partial(handle_clarify, configuration=configuration, generator_llm=generator_llm))
    graph.add_node("greeting", partial(handle_greeting, generator_llm=generator_llm))
    graph.add_node("farewell", partial(handle_farewell, generator_llm=generator_llm))

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", _route_from_state, {
        "retrieve_docs": "retrieve_docs",
        "answer_direct": "answer_direct",
        "clarify": "clarify",
        "greeting": "greeting",
        "farewell": "farewell",
    })
    graph.add_edge("retrieve_docs", "answer_with_rag")
    graph.add_edge("answer_with_rag", END)
    graph.add_edge("answer_direct", END)
    graph.add_edge("clarify", END)
    graph.add_edge("greeting", END)
    graph.add_edge("farewell", END)
    return graph.compile()
