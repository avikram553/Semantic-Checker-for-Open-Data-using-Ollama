
import pandas as pd

def calculate_metrics(df, similarity_col, threshold=0.5):
    """
    Calculate TP, FP, TN, FN, Accuracy, Precision, Recall, and F1-Score.
    
    Args:
        df: DataFrame containing 'Match' column and the similarity column.
        similarity_col: Name of the column containing similarity scores.
        threshold: Threshold for classification (default 0.5).
        
    Returns:
        Dictionary containing all metrics.
    """
    # Ensure we strictly operate on a boolean copy
    predicted = df[similarity_col] >= threshold
    
    # Handle ground truth column (case insensitive string or boolean)
    match_col = df['Match']
    
    tp = len(df[(predicted) & (match_col)])
    fp = len(df[(predicted) & (~match_col)])
    tn = len(df[(~predicted) & (~match_col)])
    fn = len(df[(~predicted) & (match_col)])
    
    total = len(df)
    accuracy = (tp + tn) / total * 100 if total > 0 else 0
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * tp / (2 * tp + fp + fn) * 100 if (2 * tp + fp + fn) > 0 else 0
    
    return {
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
        'Accuracy': accuracy, 
        'Precision': precision, 
        'Recall': recall, 
        'F1': f1
    }
