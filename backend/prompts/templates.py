"""LangChain prompt templates."""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from backend.prompts.system import SYSTEM_PROMPT, GREETING_PROMPT


def build_chat_prompt(is_greeting: bool = False) -> ChatPromptTemplate:
    """
    Build prompt. Uses MessagesPlaceholder for history so the LLM
    sees real message objects instead of flattened text.
    """
    system = GREETING_PROMPT if is_greeting else SYSTEM_PROMPT
    
    return ChatPromptTemplate.from_messages([
        ("system", system),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])