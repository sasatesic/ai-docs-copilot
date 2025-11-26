# api_service/clients/llm_client.py

from typing import List, Dict, Any
from openai import OpenAI

from api_service.config import Settings


class LLMClient:
    """
    Thin wrapper around OpenAI's chat completion API.
    Keeps all LLM-related logic in one place.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        # Standard OpenAI client
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """
        Synchronous helper for simple chat calls.
        For this project that's enough; if needed we can add async later.
        """
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            **kwargs,
        )
        # new OpenAI client returns `choices[0].message.content`
        return response.choices[0].message.content or ""
