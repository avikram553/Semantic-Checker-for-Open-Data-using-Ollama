"""
Ollama Client for interacting with local LLM
"""

import requests
from typing import Dict, Any, Optional
import json


class OllamaClient:
    """Client for communicating with Ollama API."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.1:8b"):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Ollama API base URL
            model: Model name to use (default: llama3.1:8b)
        """
        self.base_url = base_url
        self.model = model
        self.api_url = f"{base_url}/api"
    
    def is_available(self) -> bool:
        """
        Check if Ollama service is available.
        
        Returns:
            bool: True if Ollama is running, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                temperature: float = 0.7, max_tokens: int = 512) -> Dict[str, Any]:
        """
        Generate response from Ollama.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dict containing response and metadata
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            response = requests.post(
                f"{self.api_url}/generate",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "response": ""}
    
    def chat(self, messages: list, temperature: float = 0.7) -> Dict[str, Any]:
        """
        Chat completion with Ollama.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature
            
        Returns:
            Dict containing response and metadata
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/chat",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "message": {"content": ""}}
    
    def embeddings(self, text: str) -> Optional[list]:
        """
        Get embeddings for text.
        
        Args:
            text: Input text
            
        Returns:
            List of embedding values or None if error
        """
        payload = {
            "model": self.model,
            "prompt": text
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/embeddings",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json().get("embedding")
        except requests.exceptions.RequestException:
            return None
