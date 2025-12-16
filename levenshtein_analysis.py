"""
Levenshtein Distance Analysis on Ground Truth Test Pairs
"""

import json
import pandas as pd
from pathlib import Path
import numpy as np


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate the Levenshtein distance between two strings.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        int: Levenshtein distance (minimum number of single-character edits)
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def normalized_levenshtein(s1: str, s2: str) -> float:
    """
    Calculate normalized Levenshtein distance (0 to 1).
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        float: Normalized distance (0 = identical, 1 = completely different)
    """
    distance = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    
    if max_len == 0:
        return 0.0
    
    return distance / max_len


def levenshtein_similarity(s1: str, s2: str) -> float:
    """
    Calculate Levenshtein similarity score (0 to 1).
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        float: Similarity score (1 = identical, 0 = completely different)
    """
    return 1.0 - normalized_levenshtein(s1, s2)


def analyze_pairs(data_file: str, output_csv: str = "data/levenshtein_results.csv"):
    """
    Analyze ground truth test pairs using Levenshtein distance.
    
    Args:
        data_file: Path to ground truth JSON file
        output_csv: Path to save results CSV
    """
    print("=" * 80)
    print("LEVENSHTEIN DISTANCE ANALYSIS")
    print("Ground Truth Test Pairs")
    print("=" * 80)
    print()
    
    # Load the data
    print(f"Loading data from: {data_file}")
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    metadata = data.get('metadata', {})
    print(f"Dataset: {metadata.get('title', 'Unknown')}")
    print(f"Total pairs: {metadata.get('total_pairs', 0)}")
    print()
    
    # Process all test pairs
    results = []
    
    print("Processing test pairs...")
    for pair in data.get('test_pairs', []):
        attr1 = pair['attribute1']
        attr2 = pair['attribute2']
        
        # Calculate Levenshtein metrics (case-insensitive)
        lev_dist = levenshtein_distance(attr1.lower(), attr2.lower())
        norm_dist = normalized_levenshtein(attr1.lower(), attr2.lower())
        similarity = levenshtein_similarity(attr1.lower(), attr2.lower())
        
        # Also calculate case-sensitive metrics
        lev_dist_case = levenshtein_distance(attr1, attr2)
        similarity_case = levenshtein_similarity(attr1, attr2)
        
        results.append({
            'id': pair['id'],
            'attribute1': attr1,
            'attribute2': attr2,
            'match': pair['match'],
            'confidence': pair.get('confidence', 'unknown'),
            'category': pair.get('category', 'unknown'),
            'type': pair.get('type', 'unknown'),
            'reasoning': pair.get('reasoning', ''),
            'levenshtein_distance': lev_dist,
            'normalized_distance': round(norm_dist, 4),
            'similarity_score': round(similarity, 4),
            'lev_dist_case_sensitive': lev_dist_case,
            'similarity_case_sensitive': round(similarity_case, 4),
            'length_attr1': len(attr1),
            'length_attr2': len(attr2),
            'max_length': max(len(attr1), len(attr2))
        })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by ID
    df = df.sort_values('id').reset_index(drop=True)
    
    # Save to CSV
    print(f"\nSaving results to: {output_csv}")
    df.to_csv(output_csv, index=False, encoding='utf-8')
    
    # Display statistics
    print("\n" + "=" * 80)
    print("LEVENSHTEIN DISTANCE STATISTICS")
    print("=" * 80)
    print()
    
    # Overall statistics
    print("Overall Statistics (Case-Insensitive):")
    print(f"  Total pairs analyzed: {len(df)}")
    print(f"  Average Levenshtein distance: {df['levenshtein_distance'].mean():.2f}")
    print(f"  Average normalized distance: {df['normalized_distance'].mean():.4f}")
    print(f"  Average similarity score: {df['similarity_score'].mean():.4f}")
    print(f"  Median similarity score: {df['similarity_score'].median():.4f}")
    print(f"  Std dev similarity score: {df['similarity_score'].std():.4f}")
    print()
    
    # Statistics by match status
    print("Statistics by Match Status:")
    for match_status in [True, False]:
        subset = df[df['match'] == match_status]
        label = "MATCHING PAIRS (True)" if match_status else "NON-MATCHING PAIRS (False)"
        print(f"\n  {label}:")
        print(f"    Count: {len(subset)}")
        print(f"    Avg Levenshtein distance: {subset['levenshtein_distance'].mean():.2f}")
        print(f"    Avg normalized distance: {subset['normalized_distance'].mean():.4f}")
        print(f"    Avg similarity score: {subset['similarity_score'].mean():.4f}")
        print(f"    Min similarity: {subset['similarity_score'].min():.4f}")
        print(f"    Max similarity: {subset['similarity_score'].max():.4f}")
        print(f"    Median similarity: {subset['similarity_score'].median():.4f}")
    
    # Statistics by category
    print("\n\nStatistics by Category:")
    category_stats = df.groupby('category').agg({
        'similarity_score': ['count', 'mean', 'std', 'min', 'max'],
        'levenshtein_distance': 'mean'
    }).round(4)
    print(category_stats)
    
    # Statistics by type
    print("\n\nStatistics by Type:")
    type_stats = df.groupby('type').agg({
        'similarity_score': ['count', 'mean', 'std'],
        'levenshtein_distance': 'mean'
    }).round(4)
    print(type_stats)
    
    # Statistics by confidence
    print("\n\nStatistics by Confidence:")
    confidence_stats = df.groupby('confidence').agg({
        'similarity_score': ['count', 'mean', 'std'],
        'match': 'sum'
    }).round(4)
    print(confidence_stats)
    
    # Find interesting examples
    print("\n\n" + "=" * 80)
    print("TOP 10 HIGHEST SIMILARITY (by Levenshtein)")
    print("=" * 80)
    top_similar = df.nlargest(10, 'similarity_score')[['attribute1', 'attribute2', 'match', 'similarity_score', 'category', 'type']]
    print(top_similar.to_string(index=False))
    
    print("\n\n" + "=" * 80)
    print("TOP 10 LOWEST SIMILARITY (by Levenshtein)")
    print("=" * 80)
    top_dissimilar = df.nsmallest(10, 'similarity_score')[['attribute1', 'attribute2', 'match', 'similarity_score', 'category', 'type']]
    print(top_dissimilar.to_string(index=False))
    
    # Challenging cases - matching pairs with low similarity
    print("\n\n" + "=" * 80)
    print("CHALLENGING MATCHES (True matches with LOW Levenshtein similarity)")
    print("=" * 80)
    challenging = df[(df['match'] == True) & (df['similarity_score'] < 0.5)].nsmallest(10, 'similarity_score')
    if len(challenging) > 0:
        print(challenging[['attribute1', 'attribute2', 'similarity_score', 'category', 'reasoning']].to_string(index=False))
    else:
        print("No challenging matches found (all true matches have similarity >= 0.5)")
    
    # False friends - non-matching pairs with high similarity
    print("\n\n" + "=" * 80)
    print("FALSE FRIENDS (Non-matches with HIGH Levenshtein similarity)")
    print("=" * 80)
    false_friends = df[(df['match'] == False) & (df['similarity_score'] > 0.5)].nlargest(10, 'similarity_score')
    if len(false_friends) > 0:
        print(false_friends[['attribute1', 'attribute2', 'similarity_score', 'category', 'reasoning']].to_string(index=False))
    else:
        print("No false friends found (all non-matches have similarity <= 0.5)")
    
    # Threshold analysis for classification
    print("\n\n" + "=" * 80)
    print("THRESHOLD ANALYSIS FOR CLASSIFICATION")
    print("=" * 80)
    print("\nUsing Levenshtein similarity to predict matches:")
    print()
    
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print(f"{'Threshold':<12} {'TP':<6} {'FP':<6} {'TN':<6} {'FN':<6} {'Accuracy':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 100)
    
    for threshold in thresholds:
        # Predict: if similarity >= threshold, predict match=True
        df['predicted'] = df['similarity_score'] >= threshold
        
        tp = len(df[(df['predicted'] == True) & (df['match'] == True)])
        fp = len(df[(df['predicted'] == True) & (df['match'] == False)])
        tn = len(df[(df['predicted'] == False) & (df['match'] == False)])
        fn = len(df[(df['predicted'] == False) & (df['match'] == True)])
        
        accuracy = (tp + tn) / len(df) if len(df) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{threshold:<12.1f} {tp:<6} {fp:<6} {tn:<6} {fn:<6} {accuracy:<10.4f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")
    
    # Find optimal threshold
    best_f1 = 0
    best_threshold = 0
    for threshold in np.arange(0.1, 1.0, 0.01):
        df['predicted'] = df['similarity_score'] >= threshold
        tp = len(df[(df['predicted'] == True) & (df['match'] == True)])
        fp = len(df[(df['predicted'] == True) & (df['match'] == False)])
        fn = len(df[(df['predicted'] == False) & (df['match'] == True)])
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print("\n" + "-" * 100)
    print(f"Optimal threshold: {best_threshold:.2f} (F1-Score: {best_f1:.4f})")
    
    print("\n" + "=" * 80)
    print(f"Analysis complete! Results saved to: {output_csv}")
    print("=" * 80)
    
    return df


def main():
    """Main function to run Levenshtein analysis."""
    data_file = "data/ground_truth_300_pairs.json"
    output_file = "data/levenshtein_results.csv"
    
    if not Path(data_file).exists():
        print(f"Error: {data_file} not found!")
        return
    
    df = analyze_pairs(data_file, output_file)
    
    print("\n✅ Levenshtein distance analysis complete!")
    print(f"📊 Total pairs analyzed: {len(df)}")
    print(f"💾 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
