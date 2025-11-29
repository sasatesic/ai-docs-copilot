# tests/test_llm_client.py

import pytest
import asyncio
from unittest.mock import AsyncMock, patch

from api_service.clients.llm_client import LLMClient
from api_service.config import Settings

# Define a mock response object structure that matches AsyncOpenAI's completion response
class MockCompletionMessage:
    def __init__(self, content):
        self.content = content

class MockCompletionChoice:
    def __init__(self, content):
        self.message = MockCompletionMessage(content)

class MockCompletionResponse:
    def __init__(self, content):
        self.choices = [MockCompletionChoice(content)]

@pytest.fixture
def mock_settings():
    """Provides a minimal Settings object needed for client initialization."""
    return Settings(openai_api_key="dummy-key", openai_model="gpt-4-test")


@pytest.fixture
def mock_openai_client():
    """Mocks the internal AsyncOpenAI client instance used by LLMClient."""
    mock_client = AsyncMock()
    
    # Configure the mock response for the chat completion call
    mock_response = MockCompletionResponse("The AI generated this answer.")
    mock_client.chat.completions.create.return_value = mock_response
    
    return mock_client


# Patch the external AsyncOpenAI class itself to inject our mock client
@patch("api_service.clients.llm_client.AsyncOpenAI")
def test_llm_client_chat_call_is_correct(MockAsyncOpenAI, mock_settings, mock_openai_client):
    """
    Tests that the chat method correctly initializes the client and calls the 
    underlying API method with the right arguments.
    """
    # 1. Inject our mock client instance when LLMClient tries to initialize AsyncOpenAI
    MockAsyncOpenAI.return_value = mock_openai_client

    # 2. Instantiate the LLMClient
    client = LLMClient(mock_settings)
    
    # 3. Define test inputs
    test_messages = [
        {"role": "system", "content": "You are a tester."},
        {"role": "user", "content": "What is the test question?"},
    ]
    test_kwargs = {"temperature": 0.5, "max_tokens": 512}
    
    # 4. Run the async method
    response = asyncio.run(client.chat(test_messages, **test_kwargs))

    # 5. Assertions

    # Verify the underlying API method was called once
    mock_openai_client.chat.completions.create.assert_called_once()
    
    # Verify arguments passed to the API call
    args, kwargs = mock_openai_client.chat.completions.create.call_args
    
    assert kwargs["model"] == "gpt-4-test"
    assert kwargs["messages"] == test_messages
    assert kwargs["temperature"] == 0.5
    assert kwargs["max_tokens"] == 512
    
    # Verify the method returns the content from the mock response
    assert response == "The AI generated this answer."

@patch("api_service.clients.llm_client.AsyncOpenAI")
def test_llm_client_initialization(MockAsyncOpenAI, mock_settings):
    """
    Tests that the LLMClient initializes AsyncOpenAI with the correct key.
    """
    # Instantiate the LLMClient
    LLMClient(mock_settings)
    
    # Verify the AsyncOpenAI class was initialized with the correct key from settings
    MockAsyncOpenAI.assert_called_once_with(api_key="dummy-key")