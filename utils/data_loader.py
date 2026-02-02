
import pandas as pd
import json
from pathlib import Path

def load_pair_data(filepath: str) -> pd.DataFrame:
    """
    Load attribute pairs from CSV or JSON file into a unified DataFrame.
    
    Args:
        filepath: Path to the input file.
        
    Returns:
        DataFrame with columns ['ID', 'Attribute1', 'Attribute2', 'Match', ...]
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
        
    if path.suffix.lower() == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Parse standard JSON format used in this project
        pairs = data.get('test_pairs', [])
        df = pd.DataFrame(pairs)
        
        # Standardize column names if needed (JSON uses lowercase usually)
        # We want Title Case for consistency with CSV if that's the standard, 
        # or we accept whatever is there. 
        # Let's align to the CSV columns: ID, Attribute1, Attribute2, Match
        
        rename_map = {
            'id': 'ID',
            'attribute1': 'Attribute1',
            'attribute2': 'Attribute2',
            'match': 'Match',
            'confidence': 'Confidence',
            'category': 'Category', 
            'type': 'Type',
            'reasoning': 'Reasoning'
        }
        df = df.rename(columns=rename_map)
        
    elif path.suffix.lower() == '.csv':
        df = pd.read_csv(path, encoding='utf-8')
        
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
        
    # Ensure Match column is boolean
    if 'Match' in df.columns:
        if df['Match'].dtype == 'object':
             df['Match'] = df['Match'].astype(str).str.lower() == 'true'
             
    return df
