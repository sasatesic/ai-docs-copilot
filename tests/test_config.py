# tests/test_config.py

import pytest
from api_service.config import Settings, get_settings

# Reset the cache before each test that manipulates settings
@pytest.fixture(autouse=True)
def cleanup_settings_cache():
    get_settings.cache_clear()
    yield

def test_settings_load_required_openai_key(monkeypatch):
    """
    Test that Settings loads the mandatory openai_api_key from environment.
    """
    # 1. Set required environment variables
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key-123")
    
    # 2. Use a dummy key for Cohere, which is optional but defined in the model
    monkeypatch.setenv("COHERE_API_KEY", "test-cohere-key-456") 
    
    # 3. Instantiate Settings
    settings = Settings()
    
    # 4. Assertions
    assert settings.openai_api_key == "test-openai-key-123"

def test_settings_uses_default_values(monkeypatch):
    """
    Test that optional settings use their default values when not provided.
    """
    # 1. Set only the mandatory key
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key-123")
    # 2. Ensure optional keys are NOT set in the environment
    monkeypatch.delenv("QDRANT_HOST", raising=False)
    monkeypatch.delenv("APP_ENV", raising=False)
    
    # 3. Instantiate Settings
    settings = Settings()
    
    # 4. Assertions (check defaults defined in config.py)
    assert settings.openai_model == "gpt-4.1-mini"
    assert settings.qdrant_host == "localhost"
    assert settings.app_env == "local"

def test_get_settings_is_cached(monkeypatch):
    """
    Test that get_settings uses caching (lru_cache) and is only initialized once.
    """
    # 1. Set environment variable
    monkeypatch.setenv("OPENAI_API_KEY", "first-load-key")
    
    # 2. Load settings for the first time
    settings1 = get_settings()
    
    # 3. Change environment variable (should be ignored due to cache)
    monkeypatch.setenv("OPENAI_API_KEY", "second-load-key")
    
    # 4. Load settings for the second time
    settings2 = get_settings()
    
    # 5. Assertions: the key should remain the value from the first load
    assert settings1 is settings2 # Should be the exact same object
    assert settings2.openai_api_key == "first-load-key"