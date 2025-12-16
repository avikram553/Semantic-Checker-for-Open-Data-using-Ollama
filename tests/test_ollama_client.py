"""
Unit tests for OllamaClient
"""

import pytest
from src.ollama_client import OllamaClient


def test_ollama_client_initialization():
    """Test OllamaClient initialization."""
    client = OllamaClient()
    assert client.base_url == "http://localhost:11434"
    assert client.model == "llama3.1:8b"


def test_ollama_client_custom_model():
    """Test OllamaClient with custom model."""
    client = OllamaClient(model="llama3")
    assert client.model == "llama3"


# Add more tests as needed
