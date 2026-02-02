

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from transformers import BertModel, BertTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to allow importing utils
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.data_loader import load_pair_data
from utils.evaluation import calculate_metrics

def load_bert_model(model_name: str = 'bert-base-uncased'):
    
    print(f"Loading BERT model: {model_name}")
    print("This may take a moment on first run (downloads model)...")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()
    print("✅ Model loaded successfully!")
    return tokenizer, model

def get_cls_embedding(text: str, tokenizer, model):
    
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    
    return cls_embedding

def calculate_bert_similarity(attr1: str, attr2: str, tokenizer, model) -> float:
    
    emb1 = get_cls_embedding(attr1, tokenizer, model)
    emb2 = get_cls_embedding(attr2, tokenizer, model)
    
    similarity = cosine_similarity([emb1], [emb2])[0][0]
    
    normalized_similarity = (similarity + 1) / 2
    
    return normalized_similarity

def analyze_pairs(data_file: str, output_csv: str = "data/bert_base_results.csv", 
                 model_name: str = 'bert-base-uncased'):
    
    print("=" * 80)
    print("BERT BASE (ENGLISH-ONLY) SEMANTIC SIMILARITY ANALYSIS")
    print("Ground Truth Test Pairs")
    print("=" * 80)
    print()
    
    tokenizer, model = load_bert_model(model_name)
    print()
    
    print(f"Loading data from: {data_file}")
    
    try:
        df = load_pair_data(data_file)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
        
    print(f"Dataset: Ground Truth Pairs")
    print(f"Total pairs: {len(df)}")
    print()
    
    results = []
    print("Processing pairs...")
    print("-" * 80)
    
    total_pairs = len(df)
    
    for i, row in df.iterrows():
        attr1 = str(row['Attribute1'])
        attr2 = str(row['Attribute2'])
        match = row['Match']
        
        bert_sim = calculate_bert_similarity(attr1, attr2, tokenizer, model)
        
        result = {
            'id': row['ID'],
            'attribute1': attr1,
            'attribute2': attr2,
            'match': match,
            'confidence': row.get('Confidence', 'unknown'),
            'category': row.get('Category', 'unknown'),
            'type': row.get('Type', 'unknown'),
            'reasoning': row.get('Reasoning', ''),
            'bert_base_similarity': bert_sim,
            'length_attr1': len(attr1),
            'length_attr2': len(attr2),
            'max_length': max(len(attr1), len(attr2))
        }
        
        results.append(result)
        
        if (i + 1) % 50 == 0 or (i + 1) == total_pairs:
            print(f"Processed {i + 1}/{total_pairs} pairs...")
    
    print(f"Processed all {len(results)} pairs!")
    print()
    
    df = pd.DataFrame(results)
    
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    matches = df[df['match'] == True]
    non_matches = df[df['match'] == False]
    
    print(f"\nTotal pairs: {len(df)}")
    print(f"Matching pairs: {len(matches)} ({len(matches)/len(df)*100:.1f}%)")
    print(f"Non-matching pairs: {len(non_matches)} ({len(non_matches)/len(df)*100:.1f}%)")
    
    print(f"\n{'Metric':<30} {'All Pairs':<15} {'Matches':<15} {'Non-Matches':<15}")
    print("-" * 75)
    
    metrics = [
        ('BERT Base Similarity', 'bert_base_similarity'),
    ]
    
    for metric_name, col in metrics:
        all_avg = df[col].mean()
        match_avg = matches[col].mean()
        non_match_avg = non_matches[col].mean()
        
        print(f"{metric_name:<30} {all_avg:>14.4f} {match_avg:>14.4f} {non_match_avg:>14.4f}")
    
    print("\n" + "=" * 80)
    print("THRESHOLD ANALYSIS")
    print("=" * 80)
    
    # Use metrics utility for this one too? Or just keep as is?
    # Keeping it as is for now for visual consistency with original script, 
    # but calculating best F1 is also done in utils. 
    # Let's use utils for the Best F1 calculation at least to show we use it.
    
    metrics_result = calculate_metrics(df, 'bert_base_similarity', threshold=0.5)
    
    print(f"Standard Metrics (Threshold 0.5): F1: {metrics_result['F1']:.2f}%, Accuracy: {metrics_result['Accuracy']:.2f}%")
    print()
    
    thresholds = np.arange(0.5, 1.0, 0.05)
    
    best_f1 = 0
    best_threshold = 0
    
    print(f"\n{'Threshold':<12} {'TP':<6} {'FP':<6} {'TN':<6} {'FN':<6} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 90)
    
    for threshold in thresholds:
        df['predicted_match'] = df['bert_base_similarity'] >= threshold
        
        tp = len(df[(df['predicted_match'] == True) & (df['match'] == True)])
        fp = len(df[(df['predicted_match'] == True) & (df['match'] == False)])
        tn = len(df[(df['predicted_match'] == False) & (df['match'] == False)])
        fn = len(df[(df['predicted_match'] == False) & (df['match'] == True)])
        
        accuracy = (tp + tn) / len(df) * 100
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
        
        print(f"{threshold:<12.2f} {tp:<6} {fp:<6} {tn:<6} {fn:<6} {accuracy:<10.2f} {precision:<10.2f} {recall:<10.2f} {f1:<10.2f}")
    
    print("\n" + "=" * 80)
    print(f"Best F1-Score: {best_f1:.2f}% at threshold {best_threshold:.2f}")
    print("=" * 80)
    
    if 'predicted_match' in df.columns:
        df = df.drop('predicted_match', axis=1)
    
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to: {output_path}")
    
    print("\n" + "=" * 80)
    print("CATEGORY BREAKDOWN")
    print("=" * 80)
    
    for category in sorted(df['category'].unique()):
        cat_data = df[df['category'] == category]
        cat_matches = cat_data[cat_data['match'] == True]
        cat_non_matches = cat_data[cat_data['match'] == False]
        
        print(f"\n{category.upper()}")
        print(f"  Total pairs: {len(cat_data)}")
        print(f"  Matches: {len(cat_matches)}, Non-matches: {len(cat_non_matches)}")
        print(f"  Avg similarity (matches): {cat_matches['bert_base_similarity'].mean():.4f}")
        if len(cat_non_matches) > 0:
            print(f"  Avg similarity (non-matches): {cat_non_matches['bert_base_similarity'].mean():.4f}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "data" / "clean_ground_truth_1000.csv"
    default_output = script_dir / "data" / "bert_base_results.csv"

    parser = argparse.ArgumentParser(description='Run BERT analysis')
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
            print(f"❌ Error: Input file not found: {args.input}")
            exit(1)
    
    analyze_pairs(str(input_path), args.output)
    print("\n✅ BERT Base analysis complete!")
