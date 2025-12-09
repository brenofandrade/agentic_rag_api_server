import logging
import os
from typing import Any, Dict, List

from flask import Flask, jsonify, request
from flask_cors import CORS
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

import config  # loads .env into process env
from src.retrieval_graph.graph import AgentState, build_graph

logger = logging.getLogger(__name__)


def _load_graph():
    # Build once at startup so we reuse LLM/vector store clients
    return build_graph()


def _messages_from_history(history: List[Dict[str, Any]]) -> List[BaseMessage]:
    messages: List[BaseMessage] = []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role")
        content = entry.get("content")
        if content is None:
            continue
        if role == "user":
            messages.append(HumanMessage(content=str(content)))
        elif role == "assistant":
            messages.append(AIMessage(content=str(content)))
    return messages


def create_app() -> Flask:
    graph = _load_graph()

    app = Flask(__name__)
    CORS(app)

    @app.route("/health", methods=["GET"])
    def health_check():
        return jsonify({"status": "ok"}), 200

    @app.route("/chat", methods=["POST"])
    def chat():
        data = request.get_json(force=True) or {}
        raw_question = data.get("question")

        if raw_question is None or not str(raw_question).strip():
            return jsonify({"error": "Campo 'question' é obrigatório."}), 400

        question: str = str(raw_question).strip()
        history = data.get("history", []) or []

        messages = _messages_from_history(history)

        initial_state: AgentState = {
            "question": question,
            "messages": messages,
            "docs": [],
            "route": None,
            "answer": None,
            "retrieval_error": None,
        }

        logger.info("[CHAT] Pergunta recebida: %s", question)
        final_state = graph.invoke(initial_state)

        answer = final_state.get("answer")
        route = final_state.get("route")

        route_label = route.get("label") if isinstance(route, dict) else route
        router_message = route.get("message_to_user") if isinstance(route, dict) else None

        logger.info("[CHAT] Rota: %s | Resposta: %s", route_label, answer)

        return jsonify(
            {
                "answer": answer,
                "route": route_label,
                "router_message": router_message,
            }
        )

    return app


if __name__ == "__main__":
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    app = create_app()
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)
