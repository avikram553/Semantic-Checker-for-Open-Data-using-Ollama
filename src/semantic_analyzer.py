"""
Semantic Analyzer using Ollama LLM
"""

from typing import List, Dict, Any
from src.ollama_client import OllamaClient
from tqdm import tqdm


class SemanticAnalyzer:
    """Analyzes semantic similarity using Ollama LLM."""
    
    def __init__(self, ollama_client: OllamaClient):
        """
        Initialize semantic analyzer.
        
        Args:
            ollama_client: OllamaClient instance
        """
        self.client = ollama_client
        self.system_prompt = """You are a semantic similarity expert. 
        Analyze the semantic relationship between two text inputs.
        Determine if they convey the same meaning, even if worded differently.
        Respond with 'SIMILAR' or 'DIFFERENT' followed by a brief explanation."""
    
    def analyze_pair(self, text1: str, text2: str) -> Dict[str, Any]:
        """
        Analyze semantic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Dict with similarity result and explanation
        """
        prompt = f"""Compare these two texts:

Text 1: {text1}
Text 2: {text2}

Are these semantically similar (same meaning)?
Answer with SIMILAR or DIFFERENT, then explain why."""
        
        result = self.client.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=0.3
        )
        
        if "error" in result:
            return {
                "text1": text1,
                "text2": text2,
                "prediction": "ERROR",
                "explanation": result.get("error"),
                "confidence": 0.0
            }
        
        response = result.get("response", "")
        prediction = "SIMILAR" if "SIMILAR" in response.upper() else "DIFFERENT"
        
        return {
            "text1": text1,
            "text2": text2,
            "prediction": prediction,
            "explanation": response,
            "confidence": self._extract_confidence(response)
        }
    
    def analyze_batch(self, data: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Analyze multiple text pairs.
        
        Args:
            data: List of dicts with 'text1' and 'text2' keys
            
        Returns:
            List of analysis results
        """
        results = []
        for item in tqdm(data, desc="Analyzing pairs"):
            result = self.analyze_pair(item['text1'], item['text2'])
            if 'label' in item:
                result['true_label'] = item['label']
            results.append(result)
        
        return results
    
    def _extract_confidence(self, response: str) -> float:
        """
        Extract confidence score from response.
        
        Args:
            response: LLM response text
            
        Returns:
            Confidence score between 0 and 1
        """
        # Simple heuristic: look for confidence indicators
        confidence_keywords = {
            'definitely': 0.95,
            'clearly': 0.9,
            'likely': 0.75,
            'probably': 0.7,
            'possibly': 0.6,
            'maybe': 0.5
        }
        
        response_lower = response.lower()
        for keyword, score in confidence_keywords.items():
            if keyword in response_lower:
                return score
        
        return 0.8  # Default confidence
