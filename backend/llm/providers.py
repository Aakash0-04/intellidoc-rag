"""LLM provider factory for OpenAI, Google, Groq, OpenRouter."""
from __future__ import annotations

from typing import Literal, Optional

from backend.config.settings import get_settings
from backend.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class LLMFactory:
    """Create LLM instances for any supported provider."""
    
    @staticmethod
    def create(
        provider: Optional[Literal["openai", "google", "groq", "openrouter"]] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
    ):
        provider = provider or settings.default_llm_provider
        model = model or settings.default_llm_model
        api_key = settings.get_llm_key(provider)
        
        if not api_key:
            raise ValueError(f"No API key for provider: {provider}")
        
        logger.info(f"Creating LLM: {provider}/{model}")
        
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=model, temperature=temperature, api_key=api_key)
        
        elif provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=api_key)
        
        elif provider == "groq":
            from langchain_groq import ChatGroq
            return ChatGroq(model_name=model, temperature=temperature, api_key=api_key)
        
        elif provider == "openrouter":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
            )
        
        raise ValueError(f"Unknown provider: {provider}")