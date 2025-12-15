"""
Data Processor for loading and saving CSV data
"""

import pandas as pd
from typing import List, Dict, Any
from pathlib import Path


class DataProcessor:
    """Handles data loading and saving operations."""
    
    def __init__(self, input_path: str):
        """
        Initialize data processor.
        
        Args:
            input_path: Path to input CSV file
        """
        self.input_path = Path(input_path)
        self.data = None
        self.load_data()
    
    def load_data(self) -> None:
        """Load data from CSV file."""
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        
        self.data = pd.read_csv(self.input_path)
        
        # Validate required columns
        required_columns = ['text1', 'text2']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        print(f"Loaded {len(self.data)} rows from {self.input_path}")
    
    def get_data(self) -> List[Dict[str, str]]:
        """
        Get data as list of dictionaries.
        
        Returns:
            List of dictionaries with text pairs
        """
        return self.data.to_dict('records')
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str) -> None:
        """
        Save results to CSV file.
        
        Args:
            results: List of result dictionaries
            output_path: Path to output CSV file
        """
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(results)} results to {output_path}")
    
    def create_sample_data(self, output_path: str, num_samples: int = 10) -> None:
        """
        Create sample data for testing.
        
        Args:
            output_path: Path to save sample data
            num_samples: Number of sample pairs to generate
        """
        sample_data = {
            'text1': [
                "The cat sat on the mat",
                "I love programming in Python",
                "The weather is beautiful today",
                "Machine learning is fascinating",
                "Coffee helps me focus",
                "The movie was entertaining",
                "Exercise is important for health",
                "Reading books expands knowledge",
                "Technology changes rapidly",
                "Music brings people together"
            ][:num_samples],
            'text2': [
                "A feline rested on the rug",
                "Python programming is my passion",
                "Today's weather is wonderful",
                "AI and ML are very interesting",
                "I need coffee to concentrate",
                "I enjoyed watching that film",
                "Regular physical activity maintains wellness",
                "Books help you learn new things",
                "Tech innovation happens fast",
                "People connect through musical experiences"
            ][:num_samples],
            'label': [
                'SIMILAR', 'SIMILAR', 'SIMILAR', 'SIMILAR', 'SIMILAR',
                'SIMILAR', 'SIMILAR', 'SIMILAR', 'SIMILAR', 'SIMILAR'
            ][:num_samples]
        }
        
        df = pd.DataFrame(sample_data)
        df.to_csv(output_path, index=False)
        print(f"Created sample data with {num_samples} pairs at {output_path}")
