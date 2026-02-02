
import pandas as pd
from pathlib import Path

def extract_attributes_from_csv(file_path: str, limit: int = None) -> list:
    """
    Extract column headers (attributes) from a CSV file.
    
    Args:
        file_path: Path to the CSV file.
        limit: Optional limit on number of attributes to return.
        
    Returns:
        List of attribute names (strings).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
        
    # Read only the header
    try:
        # Use simple read_csv with nrows=0 to get just headers efficiently
        df = pd.read_csv(path, nrows=0, encoding='utf-8')
        headers = list(df.columns)
        
        if limit and limit > 0:
            return headers[:limit]
            
        return headers
    except Exception as e:
        print(f"Error reading CSV {file_path}: {e}")
        return []
