from typing import List, Dict, Any
from openai import AsyncOpenAI 

from api_service.config import Settings


class LLMClient:
    """
    Thin wrapper around OpenAI's chat completion API.
    Keeps all LLM-related logic in one place.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        # Use the AsyncOpenAI client
        self._client = AsyncOpenAI(api_key=settings.openai_api_key) 
        self._model = settings.openai_model

    # Make the method asynchronous
    async def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """
        Asynchronous helper for simple chat calls.
        """
        # Await the API call
        response = await self._client.chat.completions.create( 
            model=self._model,
            messages=messages,
            **kwargs,
        )
        # new OpenAI client returns `choices[0].message.content`
        return response.choices[0].message.content or ""