import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.data_loader import load_pair_data
from utils.evaluation import calculate_metrics

def load_sbert_model(model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
    
    print(f"Loading Sentence-BERT model: {model_name}")
    print("This may take a moment on first run (downloads ~100MB)...")
    model = SentenceTransformer(model_name)
    print("✅ Model loaded successfully!")
    return model

def calculate_sbert_similarity(attr1: str, attr2: str, model: SentenceTransformer) -> float:
    
    embeddings = model.encode([attr1, attr2])
    
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    normalized_similarity = (similarity + 1) / 2
    
    return normalized_similarity

def analyze_pairs(data_file: str, output_csv: str = "data/sbert_results.csv", 
                 model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
    
    print("=" * 80)
    print("SENTENCE-BERT (SBERT) SEMANTIC SIMILARITY ANALYSIS")
    print("Ground Truth Test Pairs")
    print("=" * 80)
    print()
    
    model = load_sbert_model(model_name)
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
    total_pairs = len(df)
    
    print("Processing test pairs...")
    print("(This uses transformer models, so it may take 10-30 seconds)")
    
    for i, row in df.iterrows():
        attr1 = str(row['Attribute1'])
        attr2 = str(row['Attribute2'])
        
        sbert_similarity = calculate_sbert_similarity(attr1, attr2, model)
        
        sbert_similarity_lower = calculate_sbert_similarity(attr1.lower(), attr2.lower(), model)
        
        results.append({
            'ID': row['ID'],
            'Attribute1': attr1,
            'Attribute2': attr2,
            'Match': row['Match'],
            'Confidence': row.get('Confidence', 'unknown'),
            'Category': row.get('Category', 'unknown'),
            'Type': row.get('Type', 'unknown'),
            'Reasoning': row.get('Reasoning', ''),
            'sbert_similarity': round(sbert_similarity, 4),
            'sbert_similarity_lower': round(sbert_similarity_lower, 4),
            'length_attr1': len(attr1),
            'length_attr2': len(attr2),
            'max_length': max(len(attr1), len(attr2))
        })
        
        if (i + 1) % 50 == 0 or (i + 1) == total_pairs:
            print(f"  Processed {i + 1}/{total_pairs} pairs...")
    
    df = pd.DataFrame(results)
    df = df.sort_values('ID').reset_index(drop=True)
    
    print(f"\nSaving results to: {output_csv}")
    df.to_csv(output_csv, index=False, encoding='utf-8')
    
    print("\n" + "=" * 80)
    print("SENTENCE-BERT SIMILARITY STATISTICS")
    print("=" * 80)
    print()
    
    print("Overall Statistics:")
    print(f"  Total pairs analyzed: {len(df)}")
    print(f"  Average SBERT similarity: {df['sbert_similarity'].mean():.4f}")
    print(f"  Median SBERT similarity: {df['sbert_similarity'].median():.4f}")
    print(f"  Std dev SBERT similarity: {df['sbert_similarity'].std():.4f}")
    print()
    
    print("Statistics by Match Status:")
    for match_status in [True, False]:
        subset = df[df['Match'] == match_status]
        label = "MATCHING PAIRS (True)" if match_status else "NON-MATCHING PAIRS (False)"
        print(f"\n  {label}:")
        print(f"    Count: {len(subset)}")
        if len(subset) > 0:
            print(f"    Avg SBERT similarity: {subset['sbert_similarity'].mean():.4f}")
            print(f"    Min similarity: {subset['sbert_similarity'].min():.4f}")
            print(f"    Max similarity: {subset['sbert_similarity'].max():.4f}")
            print(f"    Median similarity: {subset['sbert_similarity'].median():.4f}")
        
    match_avg = df[df['Match'] == True]['sbert_similarity'].mean()
    non_match_avg = df[df['Match'] == False]['sbert_similarity'].mean()
    gap = match_avg - non_match_avg
    print(f"\n  📊 Separation Gap: {gap:.4f} ({gap*100:.2f}%)")
    print(f"     This is how much better SBERT scores matches vs non-matches")
    
    print("\n\nStatistics by Category:")
    if 'Category' in df.columns:
        category_stats = df.groupby('Category').agg({
            'sbert_similarity': ['count', 'mean', 'std', 'min', 'max']
        }).round(4)
        print(category_stats)
    
    print("\n\nStatistics by Type:")
    if 'Type' in df.columns:
        type_stats = df.groupby('Type').agg({
            'sbert_similarity': ['count', 'mean', 'std']
        }).round(4)
        print(type_stats)
    
    print("\n\nStatistics by Confidence:")
    if 'Confidence' in df.columns:
        confidence_stats = df.groupby('Confidence').agg({
            'sbert_similarity': ['count', 'mean', 'std'],
            'Match': 'sum'
        }).round(4)
        print(confidence_stats)
    
    print("\n\n" + "=" * 80)
    print("TOP 10 HIGHEST SIMILARITY (by SBERT)")
    print("=" * 80)
    top_similar = df.nlargest(10, 'sbert_similarity')[['Attribute1', 'Attribute2', 'Match', 'sbert_similarity', 'Category', 'Type']]
    print(top_similar.to_string(index=False))
    
    print("\n\n" + "=" * 80)
    print("TOP 10 LOWEST SIMILARITY (by SBERT)")
    print("=" * 80)
    top_dissimilar = df.nsmallest(10, 'sbert_similarity')[['Attribute1', 'Attribute2', 'Match', 'sbert_similarity', 'Category', 'Type']]
    print(top_dissimilar.to_string(index=False))
    
    print("\n\n" + "=" * 80)
    print("CROSS-LINGUAL PAIRS (SBERT's strength!)")
    print("=" * 80)
    cross_lingual = df[df['Type'].str.contains('cross_lingual', na=False) & (df['Match'] == True)].nlargest(10, 'sbert_similarity')
    if len(cross_lingual) > 0:
        print(cross_lingual[['Attribute1', 'Attribute2', 'sbert_similarity', 'Category', 'Type']].to_string(index=False))
    else:
        print("No cross-lingual pairs found")
    
    print("\n\n" + "=" * 80)
    print("CHALLENGING MATCHES (True matches with LOW SBERT similarity)")
    print("=" * 80)
    challenging = df[(df['Match'] == True) & (df['sbert_similarity'] < 0.7)].nsmallest(10, 'sbert_similarity')
    if len(challenging) > 0:
        print(challenging[['Attribute1', 'Attribute2', 'sbert_similarity', 'Category', 'Reasoning']].to_string(index=False))
    else:
        print("No challenging matches found (all true matches have SBERT similarity >= 0.7)")
        print("🎉 SBERT handles everything well!")
    
    print("\n\n" + "=" * 80)
    print("FALSE FRIENDS (Non-matches with HIGH SBERT similarity)")
    print("=" * 80)
    false_friends = df[(df['Match'] == False) & (df['sbert_similarity'] > 0.7)].nlargest(10, 'sbert_similarity')
    if len(false_friends) > 0:
        print(false_friends[['Attribute1', 'Attribute2', 'sbert_similarity', 'Category', 'Reasoning']].to_string(index=False))
    else:
        print("No false friends found (all non-matches have SBERT similarity <= 0.7)")
        print("🎉 SBERT avoids false positives!")
    
    print("\n\n" + "=" * 80)
    print("THRESHOLD ANALYSIS FOR CLASSIFICATION")
    print("=" * 80)
    print("\nUsing SBERT similarity to predict matches:")
    print("(Only considering thresholds > 0.5 for meaningful classification)")
    print()
    
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    print(f"{'Threshold':<12} {'TP':<6} {'FP':<6} {'TN':<6} {'FN':<6} {'Accuracy':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 100)
    
    for threshold in thresholds:
        df['predicted'] = df['sbert_similarity'] >= threshold
        
        tp = len(df[(df['predicted'] == True) & (df['Match'] == True)])
        fp = len(df[(df['predicted'] == True) & (df['Match'] == False)])
        tn = len(df[(df['predicted'] == False) & (df['Match'] == False)])
        fn = len(df[(df['predicted'] == False) & (df['Match'] == True)])
        
        accuracy = (tp + tn) / len(df) if len(df) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{threshold:<12.2f} {tp:<6} {fp:<6} {tn:<6} {fn:<6} {accuracy:<10.4f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")
    
    best_f1 = 0
    best_threshold = 0
    for threshold in np.arange(0.5, 1.0, 0.01):
        df['predicted'] = df['sbert_similarity'] >= threshold
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
    print()
    print("📊 COMPARISON WITH BASELINES (thresholds >0.5):")
    print(f"   Levenshtein:   F1 = 0.4161 @ threshold 0.50")
    print(f"   Jaro-Winkler:  F1 = 0.7847 @ threshold 0.50")
    print(f"   SBERT:         F1 = {best_f1:.4f} @ threshold {best_threshold:.2f}")
    
    if best_f1 > 0.7847:
        print(f"\n   🎉 SBERT WINS! ({((best_f1 - 0.7847) / 0.7847 * 100):.1f}% improvement over best baseline)")
    
    print("\n" + "=" * 80)
    print(f"Analysis complete! Results saved to: {output_csv}")
    print("=" * 80)
    
    return df

def main():
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "data" / "clean_ground_truth_1000.csv"
    default_output = script_dir / "data" / "sbert_results.csv"

    parser = argparse.ArgumentParser(description='Run SBERT analysis')
    parser.add_argument('--input', type=str, default=str(default_input), help='Path to input')
    parser.add_argument('--output', type=str, default=str(default_output), help='Path to output')
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
    
    print("\n✅ Sentence-BERT semantic similarity analysis complete!")
    print(f"📊 Total pairs analyzed: {len(df)}")
    print(f"💾 Results saved to: {args.output}")
    print("\n💡 Key insight: SBERT understands semantic meaning across languages,")
    print("   which can help with names and identifiers that start the same way!")

if __name__ == "__main__":
    main()
