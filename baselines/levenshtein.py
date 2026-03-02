import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.string_matching import levenshtein_distance, levenshtein_similarity
from utils.data_loader import load_pair_data
from utils.evaluation import calculate_metrics

def analyze_pairs(data_file: str, output_csv: str = "data/levenshtein_results.csv"):
    print("=" * 80)
    print("LEVENSHTEIN DISTANCE ANALYSIS")
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
        
        lev_dist = levenshtein_distance(attr1.lower(), attr2.lower())
        similarity = levenshtein_similarity(attr1.lower(), attr2.lower())
        
        lev_dist_case = levenshtein_distance(attr1, attr2)
        similarity_case = levenshtein_similarity(attr1, attr2)
        
        results.append({
            'ID': row['ID'],
            'Attribute1': attr1,
            'Attribute2': attr2,
            'Match': row['Match'],
            'Confidence': row.get('Confidence', 'unknown'),
            'Category': row.get('Category', 'unknown'),
            'Type': row.get('Type', 'unknown'),
            'Reasoning': row.get('Reasoning', ''),
            'levenshtein_distance': lev_dist,
            'similarity_score': round(similarity, 4),
            'lev_dist_case_sensitive': lev_dist_case,
            'similarity_case_sensitive': round(similarity_case, 4),
            'length_attr1': len(attr1),
            'length_attr2': len(attr2),
            'max_length': max(len(attr1), len(attr2))
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values('ID').reset_index(drop=True)
    
    print(f"\nSaving results to: {output_csv}")
    df.to_csv(output_csv, index=False, encoding='utf-8')
    
    # Calculate metrics using the utility function logic (manual print here to keep format)
    # Or simply call calculate_metrics and print summary
    
    metrics = calculate_metrics(df, 'similarity_score', threshold=0.70)
    
    print("\n" + "=" * 80)
    print("LEVENSHTEIN DISTANCE STATISTICS")
    print("=" * 80)
    print()
    
    print("Overall Statistics (Case-Insensitive):")
    print(f"  Total pairs analyzed: {len(df)}")
    print(f"  Average Levenshtein distance: {df['levenshtein_distance'].mean():.2f}")
    print(f"  Average normalized distance: {df['normalized_distance'].mean():.4f}")
    print(f"  Average similarity score: {df['similarity_score'].mean():.4f}")
    print(f"  Median similarity score: {df['similarity_score'].median():.4f}")
    print(f"  Std dev similarity score: {df['similarity_score'].std():.4f}")
    print()
    
    print(f"Metrics (Threshold=0.70): F1: {metrics['F1']:.2f}%, Accuracy: {metrics['Accuracy']:.2f}%")
    
    print("Statistics by Match Status:")
    for match_status in [True, False]:
        subset = df[df['Match'] == match_status]
        label = "MATCHING PAIRS (True)" if match_status else "NON-MATCHING PAIRS (False)"
        print(f"\n  {label}:")
        print(f"    Count: {len(subset)}")
        if len(subset) > 0:
            print(f"    Avg Levenshtein distance: {subset['levenshtein_distance'].mean():.2f}")
            print(f"    Avg similarity score: {subset['similarity_score'].mean():.4f}")
            print(f"    Min similarity: {subset['similarity_score'].min():.4f}")
            print(f"    Max similarity: {subset['similarity_score'].max():.4f}")
            print(f"    Median similarity: {subset['similarity_score'].median():.4f}")
    
    print("\n\nStatistics by Category:")
    if 'Category' in df.columns:
        category_stats = df.groupby('Category').agg({
            'similarity_score': ['count', 'mean', 'std', 'min', 'max'],
            'levenshtein_distance': 'mean'
        }).round(4)
        print(category_stats)
    
    print("\n\nStatistics by Type:")
    if 'Type' in df.columns:
        type_stats = df.groupby('Type').agg({
            'similarity_score': ['count', 'mean', 'std'],
            'levenshtein_distance': 'mean'
        }).round(4)
        print(type_stats)
    
    print("\n\nStatistics by Confidence:")
    if 'Confidence' in df.columns:
        confidence_stats = df.groupby('Confidence').agg({
            'similarity_score': ['count', 'mean', 'std'],
            'Match': 'sum'
        }).round(4)
        print(confidence_stats)
    
    print("\n\n" + "=" * 80)
    print("TOP 10 HIGHEST SIMILARITY (by Levenshtein)")
    print("=" * 80)
    top_similar = df.nlargest(10, 'similarity_score')[['Attribute1', 'Attribute2', 'Match', 'similarity_score', 'Category', 'Type']]
    print(top_similar.to_string(index=False))
    
    print("\n\n" + "=" * 80)
    print("TOP 10 LOWEST SIMILARITY (by Levenshtein)")
    print("=" * 80)
    top_dissimilar = df.nsmallest(10, 'similarity_score')[['Attribute1', 'Attribute2', 'Match', 'similarity_score', 'Category', 'Type']]
    print(top_dissimilar.to_string(index=False))
    
    print("\n\n" + "=" * 80)
    print("CHALLENGING MATCHES (True matches with LOW Levenshtein similarity)")
    print("=" * 80)
    challenging = df[(df['Match'] == True) & (df['similarity_score'] < 0.5)].nsmallest(10, 'similarity_score')
    if len(challenging) > 0:
        print(challenging[['Attribute1', 'Attribute2', 'similarity_score', 'Category', 'Reasoning']].to_string(index=False))
    else:
        print("No challenging matches found (all true matches have similarity >= 0.5)")
    
    print("\n\n" + "=" * 80)
    print("FALSE FRIENDS (Non-matches with HIGH Levenshtein similarity)")
    print("=" * 80)
    false_friends = df[(df['Match'] == False) & (df['similarity_score'] > 0.5)].nlargest(10, 'similarity_score')
    if len(false_friends) > 0:
        print(false_friends[['Attribute1', 'Attribute2', 'similarity_score', 'Category', 'Reasoning']].to_string(index=False))
    else:
        print("No false friends found (all non-matches have similarity <= 0.5)")
    
    print("\n\n" + "=" * 80)
    print("THRESHOLD ANALYSIS FOR CLASSIFICATION")
    print("=" * 80)
    print("\nUsing Levenshtein similarity to predict matches:")
    print("(Only considering thresholds > 0.5 for meaningful classification)")
    print()
    
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    print(f"{'Threshold':<12} {'TP':<6} {'FP':<6} {'TN':<6} {'FN':<6} {'Accuracy':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 100)
    
    for threshold in thresholds:
        df['predicted'] = df['similarity_score'] >= threshold
        
        tp = len(df[(df['predicted'] == True) & (df['Match'] == True)])
        fp = len(df[(df['predicted'] == True) & (df['Match'] == False)])
        tn = len(df[(df['predicted'] == False) & (df['Match'] == False)])
        fn = len(df[(df['predicted'] == False) & (df['Match'] == True)])
        
        accuracy = (tp + tn) / len(df) if len(df) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{threshold:<12.1f} {tp:<6} {fp:<6} {tn:<6} {fn:<6} {accuracy:<10.4f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")
    
    best_f1 = 0
    best_threshold = 0
    for threshold in np.arange(0.5, 1.0, 0.01):
        df['predicted'] = df['similarity_score'] >= threshold
        tp = len(df[(df['predicted'] == True) & (df['Match'] == True)])
        fp = len(df[(df['predicted'] == True) & (df['Match'] == False)])
        fn = len(df[(df['predicted'] == False) & (df['Match'] == True)])
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print("\n" + "-" * 100)
    print(f"Optimal threshold (>0.5): {best_threshold:.2f} (F1-Score: {best_f1:.4f})")
    
    print("\n" + "=" * 80)
    print(f"Analysis complete! Results saved to: {output_csv}")
    print("=" * 80)
    
    return df


def main():
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "data" / "clean_ground_truth_1000.csv"
    default_output = script_dir / "data" / "levenshtein_results.csv"

    parser = argparse.ArgumentParser(description='Run Levenshtein analysis on dataset')
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
            return
    
    df = analyze_pairs(str(input_path), args.output)
    
    print("\n✅ Levenshtein distance analysis complete!")
    print(f"📊 Total pairs analyzed: {len(df)}")
    print(f"💾 Results saved to: {args.output}")

if __name__ == "__main__":
    main()
