"""Setup parameters for the agent."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from langchain_core.runnables import RunnableConfig

from config import (
    K_RAG,
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_MODEL,
    OLLAMA_EMBED_MODEL,
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_ENV,
    PINECONE_NAMESPACE,
    ROUTER_MODEL,
)
from . import prompts


@dataclass
class AgentConfiguration:
    router_model: str = ROUTER_MODEL
    openai_api_key: Optional[str] = OPENAI_API_KEY
    generator_model: str = OLLAMA_CHAT_MODEL
    embed_model: str = OLLAMA_EMBED_MODEL
    ollama_base_url: str = OLLAMA_BASE_URL
    pinecone_api_key: Optional[str] = PINECONE_API_KEY
    pinecone_index_name: str = PINECONE_INDEX_NAME
    pinecone_env: Optional[str] = PINECONE_ENV
    pinecone_namespace: str = PINECONE_NAMESPACE
    rag_top_k: int = K_RAG
    router_system_prompt: str = prompts.ROUTER_SYSTEM_PROMPT
    response_system_prompt: str = prompts.RESPONSE_SYSTEM_PROMPT
    general_system_prompt: str = prompts.GENERAL_SYSTEM_PROMPT
    more_info_system_prompt: str = prompts.MORE_INFO_SYSTEM_PROMPT

    @classmethod
    def from_env(cls) -> "AgentConfiguration":
        return cls()

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig) -> "AgentConfiguration":
        base = cls.from_env()
        cfg = getattr(config, "configurable", None) or config.get("configurable", {}) if isinstance(config, dict) else {}
        # allow overrides in config.configurable
        return replace(base, **cfg)
