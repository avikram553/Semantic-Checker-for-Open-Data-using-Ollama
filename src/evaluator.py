"""
Evaluator for semantic checker performance
"""

from typing import List, Dict, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np


class Evaluator:
    """Evaluates semantic checker performance."""
    
    def __init__(self):
        """Initialize evaluator."""
        pass
    
    def evaluate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate predictions against true labels.
        
        Args:
            results: List of result dictionaries with predictions and true labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Filter results that have true labels
        labeled_results = [r for r in results if 'true_label' in r]
        
        if not labeled_results:
            return {
                "error": "No labeled data found for evaluation",
                "num_predictions": len(results)
            }
        
        true_labels = [r['true_label'] for r in labeled_results]
        predictions = [r['prediction'] for r in labeled_results]
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted', zero_division=0
        )
        
        cm = confusion_matrix(true_labels, predictions, 
                            labels=['SIMILAR', 'DIFFERENT'])
        
        # Calculate confidence statistics
        confidences = [r.get('confidence', 0.0) for r in labeled_results]
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm.tolist(),
            "num_samples": len(labeled_results),
            "avg_confidence": np.mean(confidences),
            "std_confidence": np.std(confidences)
        }
    
    def print_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Print evaluation metrics in a readable format.
        
        Args:
            metrics: Dictionary of evaluation metrics
        """
        if "error" in metrics:
            print(f"\nError: {metrics['error']}")
            print(f"Total predictions: {metrics.get('num_predictions', 0)}")
            return
        
        print("\n" + "="*50)
        print("EVALUATION METRICS")
        print("="*50)
        print(f"Number of samples: {metrics['num_samples']}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print(f"\nAvg Confidence: {metrics['avg_confidence']:.4f} ± {metrics['std_confidence']:.4f}")
        
        print("\nConfusion Matrix:")
        print("                 Predicted")
        print("              SIMILAR  DIFFERENT")
        cm = metrics['confusion_matrix']
        print(f"Actual SIMILAR    {cm[0][0]:6d}  {cm[0][1]:9d}")
        print(f"       DIFFERENT  {cm[1][0]:6d}  {cm[1][1]:9d}")
        print("="*50)
