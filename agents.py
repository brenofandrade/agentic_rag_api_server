from typing import List, Literal, Optional, Dict, Any
from typing_extensions import TypedDict, Annotated
from langchain_core.documents import Document
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage,
)

class AgentRouter(TypedDict):
    
    label: Literal[
        "rag",
        "general_knowledge",
        "clarify",
        "code_assistant",
        "greeting",
        "farewell"
    ]

    reason: str
    agent_message: str 

class AgentState(TypedDict):
    
    session_id: str
    question: str
    messages: List[BaseMessage]
    docs: List[Document]
    route: Optional[AgentRouter]
    answer: Optional[str]