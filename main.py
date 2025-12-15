"""
Semantic Checker - Main Application
Using Ollama with Llama 3.1 8B for semantic analysis
"""

from src.ollama_client import OllamaClient
from src.semantic_analyzer import SemanticAnalyzer
from src.data_processor import DataProcessor
from src.evaluator import Evaluator
import argparse
from pathlib import Path


def main():
    """Main entry point for the semantic checker application."""
    parser = argparse.ArgumentParser(description='Semantic Checker using Ollama')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output', type=str, default='results.csv', help='Path to output CSV file')
    parser.add_argument('--model', type=str, default='llama3.1:8b', help='Ollama model to use')
    parser.add_argument('--mode', type=str, choices=['analyze', 'evaluate', 'both'], 
                       default='both', help='Operation mode')
    
    args = parser.parse_args()
    
    # Initialize components
    print(f"Initializing Ollama client with model: {args.model}")
    ollama_client = OllamaClient(model=args.model)
    
    # Check if Ollama is running
    if not ollama_client.is_available():
        print("Error: Ollama is not running. Please start Ollama first.")
        return
    
    print(f"Loading data from: {args.input}")
    data_processor = DataProcessor(args.input)
    
    analyzer = SemanticAnalyzer(ollama_client)
    
    if args.mode in ['analyze', 'both']:
        print("Running semantic analysis...")
        results = analyzer.analyze_batch(data_processor.get_data())
        data_processor.save_results(results, args.output)
        print(f"Results saved to: {args.output}")
    
    if args.mode in ['evaluate', 'both']:
        print("Evaluating results...")
        evaluator = Evaluator()
        metrics = evaluator.evaluate(data_processor.get_data())
        evaluator.print_metrics(metrics)


if __name__ == "__main__":
    main()
