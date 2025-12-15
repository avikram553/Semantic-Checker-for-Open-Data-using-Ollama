"""
Semantic Checker package
"""

from src.ollama_client import OllamaClient
from src.semantic_analyzer import SemanticAnalyzer
from src.data_processor import DataProcessor
from src.evaluator import Evaluator

__version__ = "0.1.0"

__all__ = [
    "OllamaClient",
    "SemanticAnalyzer", 
    "DataProcessor",
    "Evaluator"
]
