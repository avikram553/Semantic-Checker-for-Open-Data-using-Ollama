

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path to allow importing utils
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.string_matching import jaro_similarity, jaro_winkler_similarity
from utils.data_loader import load_pair_data
from utils.evaluation import calculate_metrics

def analyze_pairs(data_file: str, output_csv: str = "data/jaro_winkler_results.csv"):
    
    print("=" * 80)
    print("JARO-WINKLER SIMILARITY ANALYSIS")
    print("Ground Truth Test Pairs")
    print("=" * 80)
    print()
    
    print(f"Loading data from: {data_file}")
    
    try:
        df = load_pair_data(data_file)
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()
        
    print(f"Dataset: Ground Truth Pairs")
    print(f"Total pairs: {len(df)}")
    print()
    
    results = []
    
    print("Processing test pairs...")
    for _, row in df.iterrows():
        attr1 = str(row['Attribute1'])
        attr2 = str(row['Attribute2'])
        
        jaro = jaro_similarity(attr1.lower(), attr2.lower())
        jaro_winkler = jaro_winkler_similarity(attr1.lower(), attr2.lower())
        
        jaro_case = jaro_similarity(attr1, attr2)
        jaro_winkler_case = jaro_winkler_similarity(attr1, attr2)
        
        results.append({
            'id': row['ID'],
            'attribute1': attr1,
            'attribute2': attr2,
            'match': row['Match'],
            'confidence': row.get('Confidence', 'unknown'),
            'category': row.get('Category', 'unknown'),
            'type': row.get('Type', 'unknown'),
            'reasoning': row.get('Reasoning', ''),
            'jaro_similarity': round(jaro, 4),
            'jaro_winkler_similarity': round(jaro_winkler, 4),
            'jaro_case_sensitive': round(jaro_case, 4),
            'jaro_winkler_case_sensitive': round(jaro_winkler_case, 4),
            'length_attr1': len(attr1),
            'length_attr2': len(attr2),
            'max_length': max(len(attr1), len(attr2))
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values('id').reset_index(drop=True)
    
    print(f"\nSaving results to: {output_csv}")
    df.to_csv(output_csv, index=False, encoding='utf-8')
    
    # Calculate metrics
    metrics = calculate_metrics(df, 'jaro_winkler_similarity', threshold=0.5)
    
    print("\n" + "=" * 80)
    print("JARO-WINKLER SIMILARITY STATISTICS")
    print("=" * 80)
    print()
    
    print("Overall Statistics (Case-Insensitive):")
    print(f"  Total pairs analyzed: {len(df)}")
    print(f"  Average Jaro similarity: {df['jaro_similarity'].mean():.4f}")
    print(f"  Average Jaro-Winkler similarity: {df['jaro_winkler_similarity'].mean():.4f}")
    print(f"  Median Jaro-Winkler similarity: {df['jaro_winkler_similarity'].median():.4f}")
    print(f"  Std dev Jaro-Winkler similarity: {df['jaro_winkler_similarity'].std():.4f}")
    print()
    
    print(f"Metrics (Threshold=0.5): F1: {metrics['F1']:.2f}%, Accuracy: {metrics['Accuracy']:.2f}%")
    print()
    
    print("Statistics by Match Status:")
    for match_status in [True, False]:
        subset = df[df['match'] == match_status]
        label = "MATCHING PAIRS (True)" if match_status else "NON-MATCHING PAIRS (False)"
        print(f"\n  {label}:")
        print(f"    Count: {len(subset)}")
        print(f"    Avg Jaro similarity: {subset['jaro_similarity'].mean():.4f}")
        print(f"    Avg Jaro-Winkler similarity: {subset['jaro_winkler_similarity'].mean():.4f}")
        print(f"    Min similarity: {subset['jaro_winkler_similarity'].min():.4f}")
        print(f"    Max similarity: {subset['jaro_winkler_similarity'].max():.4f}")
        print(f"    Median similarity: {subset['jaro_winkler_similarity'].median():.4f}")
    
    print("\n\nStatistics by Category:")
    category_stats = df.groupby('category').agg({
        'jaro_winkler_similarity': ['count', 'mean', 'std', 'min', 'max'],
        'jaro_similarity': 'mean'
    }).round(4)
    print(category_stats)
    
    print("\n\nStatistics by Type:")
    type_stats = df.groupby('type').agg({
        'jaro_winkler_similarity': ['count', 'mean', 'std'],
        'jaro_similarity': 'mean'
    }).round(4)
    print(type_stats)
    
    print("\n\nStatistics by Confidence:")
    confidence_stats = df.groupby('confidence').agg({
        'jaro_winkler_similarity': ['count', 'mean', 'std'],
        'match': 'sum'
    }).round(4)
    print(confidence_stats)
    
    print("\n\n" + "=" * 80)
    print("TOP 10 HIGHEST SIMILARITY (by Jaro-Winkler)")
    print("=" * 80)
    top_similar = df.nlargest(10, 'jaro_winkler_similarity')[['attribute1', 'attribute2', 'match', 'jaro_winkler_similarity', 'category', 'type']]
    print(top_similar.to_string(index=False))
    
    print("\n\n" + "=" * 80)
    print("TOP 10 LOWEST SIMILARITY (by Jaro-Winkler)")
    print("=" * 80)
    top_dissimilar = df.nsmallest(10, 'jaro_winkler_similarity')[['attribute1', 'attribute2', 'match', 'jaro_winkler_similarity', 'category', 'type']]
    print(top_dissimilar.to_string(index=False))
    
    print("\n\n" + "=" * 80)
    print("CHALLENGING MATCHES (True matches with LOW Jaro-Winkler similarity)")
    print("=" * 80)
    challenging = df[(df['match'] == True) & (df['jaro_winkler_similarity'] < 0.7)].nsmallest(10, 'jaro_winkler_similarity')
    if len(challenging) > 0:
        print(challenging[['attribute1', 'attribute2', 'jaro_winkler_similarity', 'category', 'reasoning']].to_string(index=False))
    else:
        print("No challenging matches found (all true matches have similarity >= 0.7)")
    
    print("\n\n" + "=" * 80)
    print("FALSE FRIENDS (Non-matches with HIGH Jaro-Winkler similarity)")
    print("=" * 80)
    false_friends = df[(df['match'] == False) & (df['jaro_winkler_similarity'] > 0.7)].nlargest(10, 'jaro_winkler_similarity')
    if len(false_friends) > 0:
        print(false_friends[['attribute1', 'attribute2', 'jaro_winkler_similarity', 'category', 'reasoning']].to_string(index=False))
    else:
        print("No false friends found (all non-matches have similarity <= 0.7)")
    
    print("\n\n" + "=" * 80)
    print("THRESHOLD ANALYSIS FOR CLASSIFICATION")
    print("=" * 80)
    print("\nUsing Jaro-Winkler similarity to predict matches:")
    print("(Only considering thresholds > 0.5 for meaningful classification)")
    print()
    
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    print(f"{'Threshold':<12} {'TP':<6} {'FP':<6} {'TN':<6} {'FN':<6} {'Accuracy':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 100)
    
    for threshold in thresholds:
        df['predicted'] = df['jaro_winkler_similarity'] >= threshold
        
        tp = len(df[(df['predicted'] == True) & (df['match'] == True)])
        fp = len(df[(df['predicted'] == True) & (df['match'] == False)])
        tn = len(df[(df['predicted'] == False) & (df['match'] == False)])
        fn = len(df[(df['predicted'] == False) & (df['match'] == True)])
        
        accuracy = (tp + tn) / len(df) if len(df) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{threshold:<12.2f} {tp:<6} {fp:<6} {tn:<6} {fn:<6} {accuracy:<10.4f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")
    
    best_f1 = 0
    best_threshold = 0
    for threshold in np.arange(0.5, 1.0, 0.01):
        df['predicted'] = df['jaro_winkler_similarity'] >= threshold
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
    print(f"Optimal threshold (>0.5): {best_threshold:.2f} (F1-Score: {best_f1:.4f})")
    print("Compare this with Levenshtein baseline (restricted to thresholds >0.5)")
    
    print("\n" + "=" * 80)
    print(f"Analysis complete! Results saved to: {output_csv}")
    print("=" * 80)
    
    return df

import argparse

def main():
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "data" / "clean_ground_truth_1000.csv"
    default_output = script_dir / "data" / "jaro_winkler_results.csv"

    parser = argparse.ArgumentParser(description='Run Jaro-Winkler analysis on dataset')
    parser.add_argument('--input', type=str, default=str(default_input), help='Path to input CSV')
    parser.add_argument('--output', type=str, default=str(default_output), help='Path to output CSV')
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        # Try finding it in data/ relative to CWD or script
        if (Path("data") / args.input).exists():
            input_path = Path("data") / args.input
        elif (script_dir / "data" / args.input).exists():
            input_path = script_dir / "data" / args.input
        else:
            print(f"Error: {args.input} not found!")
            print("Make sure you have the ground truth test pairs file.")
            return
    
    df = analyze_pairs(str(input_path), args.output)
    
    print("\n✅ Jaro-Winkler similarity analysis complete!")
    print(f"📊 Total pairs analyzed: {len(df)}")
    print(f"💾 Results saved to: {args.output}")
    print("\n💡 Key insight: Jaro-Winkler gives bonus points for matching prefixes,")
    print("   which can help with names and identifiers that start the same way!")

if __name__ == "__main__":
    main()
