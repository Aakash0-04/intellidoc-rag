"""Conversation memory per session."""
from __future__ import annotations

from typing import List, Literal

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


class ConversationMemory:
    """Simple in-memory chat history per session."""
    
    def __init__(self, max_messages: int = 10):
        self.histories: dict[str, List[BaseMessage]] = {}
        self.max_messages = max_messages
    
    def get_history(self, session_id: str) -> List[BaseMessage]:
        return self.histories.get(session_id, [])
    
    def add_message(self, session_id: str, role: Literal["human", "ai"], content: str):
        if session_id not in self.histories:
            self.histories[session_id] = []
        
        msg = HumanMessage(content=content) if role == "human" else AIMessage(content=content)
        self.histories[session_id].append(msg)
        
        if len(self.histories[session_id]) > self.max_messages * 2:
            self.histories[session_id] = self.histories[session_id][-self.max_messages * 2:]
    
    def clear(self, session_id: str):
        self.histories.pop(session_id, None)