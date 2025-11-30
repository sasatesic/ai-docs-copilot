# api_service/clients/llm_client.py

from typing import List, Dict, Any, AsyncGenerator
from openai import AsyncOpenAI
from api_service.config import Settings

class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model

    async def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Non-streaming helper (legacy)."""
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content or ""

    # NEW: Streaming method
    async def stream_chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> AsyncGenerator[str, None]:
        """
        Yields chunks of the response as they arrive.
        """
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=True, # Enable streaming in OpenAI
            **kwargs,
        )
        
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content