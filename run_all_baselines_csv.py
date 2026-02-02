import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def levenshtein_similarity(s1: str, s2: str) -> float:
    distance = levenshtein_distance(s1.lower(), s2.lower())
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return 1.0 - (distance / max_len)

def jaro_winkler_similarity(s1: str, s2: str) -> float:
    s1, s2 = s1.lower(), s2.lower()
    len1, len2 = len(s1), len(s2)
    
    if len1 == 0 and len2 == 0:
        return 1.0
    if len1 == 0 or len2 == 0:
        return 0.0
    
    match_window = max(len1, len2) // 2 - 1
    if match_window < 1:
        match_window = 1
    
    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0
    transpositions = 0
    
    for i in range(len1):
        start = max(0, i - match_window)
        end = min(i + match_window + 1, len2)
        
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = s2_matches[j] = True
            matches += 1
            break
    
    if matches == 0:
        return 0.0
    
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    
    jaro = (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3.0
    
    prefix = 0
    for i in range(min(len1, len2)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break
    prefix = min(4, prefix)
    
    return jaro + prefix * 0.1 * (1.0 - jaro)

def calculate_metrics(df, similarity_col, threshold=0.5):
    df['predicted'] = df[similarity_col] >= threshold
    tp = len(df[(df['predicted']) & (df['Match'])])
    fp = len(df[(df['predicted']) & (~df['Match'])])
    tn = len(df[(~df['predicted']) & (~df['Match'])])
    fn = len(df[(~df['predicted']) & (df['Match'])])
    
    accuracy = (tp + tn) / len(df) * 100 if len(df) > 0 else 0
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * tp / (2 * tp + fp + fn) * 100 if (2 * tp + fp + fn) > 0 else 0
    
    return {
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
        'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1
    }

def run_all_baselines(input_csv, output_prefix):
    print("=" * 80)
    print(f"🚀 RUNNING ALL 4 BASELINES ON: {input_csv}")
    print("=" * 80)
    
    df = pd.read_csv(input_csv)
    print(f"\n✅ Loaded {len(df)} pairs")
    print(f"   Matches: {df['Match'].sum()}")
    print(f"   Non-matches: {(~df['Match']).sum()}")
    
    results_df = df.copy()
    
    print("\n" + "=" * 80)
    print("1️⃣ LEVENSHTEIN DISTANCE")
    print("=" * 80)
    results_df['levenshtein_similarity'] = results_df.apply(
        lambda row: levenshtein_similarity(str(row['Attribute1']), str(row['Attribute2'])), axis=1
    )
    lev_metrics = calculate_metrics(results_df, 'levenshtein_similarity')
    print(f"F1-Score: {lev_metrics['F1']:.2f}%")
    print(f"Accuracy: {lev_metrics['Accuracy']:.2f}%")
    print(f"False Positives: {lev_metrics['FP']}, False Negatives: {lev_metrics['FN']}")
    
    print("\n" + "=" * 80)
    print("2️⃣ JARO-WINKLER DISTANCE")
    print("=" * 80)
    results_df['jaro_winkler_similarity'] = results_df.apply(
        lambda row: jaro_winkler_similarity(str(row['Attribute1']), str(row['Attribute2'])), axis=1
    )
    jw_metrics = calculate_metrics(results_df, 'jaro_winkler_similarity')
    print(f"F1-Score: {jw_metrics['F1']:.2f}%")
    print(f"Accuracy: {jw_metrics['Accuracy']:.2f}%")
    print(f"False Positives: {jw_metrics['FP']}, False Negatives: {jw_metrics['FN']}")
    
    print("\n" + "=" * 80)
    print("3️⃣ SBERT MULTILINGUAL")
    print("=" * 80)
    print("Loading model: paraphrase-multilingual-MiniLM-L12-v2...")
    sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print("✅ Model loaded!")
    
    print("Computing embeddings...")
    attr1_embeddings = sbert_model.encode(results_df['Attribute1'].tolist(), show_progress_bar=True)
    attr2_embeddings = sbert_model.encode(results_df['Attribute2'].tolist(), show_progress_bar=True)
    
    sbert_similarities = []
    for i in range(len(results_df)):
        sim = cosine_similarity([attr1_embeddings[i]], [attr2_embeddings[i]])[0][0]
        normalized_sim = (sim + 1) / 2
        sbert_similarities.append(normalized_sim)
    
    results_df['sbert_similarity'] = sbert_similarities
    sbert_metrics = calculate_metrics(results_df, 'sbert_similarity')
    print(f"F1-Score: {sbert_metrics['F1']:.2f}%")
    print(f"Accuracy: {sbert_metrics['Accuracy']:.2f}%")
    print(f"False Positives: {sbert_metrics['FP']}, False Negatives: {sbert_metrics['FN']}")
    
    print("\n" + "=" * 80)
    print("4️⃣ BERT BASE (ENGLISH)")
    print("=" * 80)
    print("Loading model: bert-base-uncased...")
    from transformers import BertModel, BertTokenizer
    import torch
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model.eval()
    print("✅ Model loaded!")
    
    def get_cls_embedding(text):
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    
    print("Computing embeddings using [CLS] token...")
    bert_similarities = []
    for idx, row in results_df.iterrows():
        if idx % 100 == 0:
            print(f"  Processed {idx}/{len(results_df)} pairs...")
        emb1 = get_cls_embedding(str(row['Attribute1']))
        emb2 = get_cls_embedding(str(row['Attribute2']))
        sim = cosine_similarity([emb1], [emb2])[0][0]
        normalized_sim = (sim + 1) / 2
        bert_similarities.append(normalized_sim)
    
    results_df['bert_base_similarity'] = bert_similarities
    bert_metrics = calculate_metrics(results_df, 'bert_base_similarity')
    print(f"F1-Score: {bert_metrics['F1']:.2f}%")
    print(f"Accuracy: {bert_metrics['Accuracy']:.2f}%")
    print(f"False Positives: {bert_metrics['FP']}, False Negatives: {bert_metrics['FN']}")
    
    output_file = f"{output_prefix}_all_baselines_results.csv"
    results_df.to_csv(output_file, index=False)
    
    print("\n" + "=" * 80)
    print("📊 FINAL COMPARISON")
    print("=" * 80)
    print(f"\n{'Method':<25} {'F1-Score':<12} {'Accuracy':<12} {'FP':<8} {'FN':<8}")
    print("-" * 70)
    print(f"{'Levenshtein':<25} {lev_metrics['F1']:>10.2f}% {lev_metrics['Accuracy']:>10.2f}% {lev_metrics['FP']:>6} {lev_metrics['FN']:>6}")
    print(f"{'Jaro-Winkler':<25} {jw_metrics['F1']:>10.2f}% {jw_metrics['Accuracy']:>10.2f}% {jw_metrics['FP']:>6} {jw_metrics['FN']:>6}")
    print(f"{'SBERT Multilingual':<25} {sbert_metrics['F1']:>10.2f}% {sbert_metrics['Accuracy']:>10.2f}% {sbert_metrics['FP']:>6} {sbert_metrics['FN']:>6}")
    print(f"{'BERT Base (English)':<25} {bert_metrics['F1']:>10.2f}% {bert_metrics['Accuracy']:>10.2f}% {bert_metrics['FP']:>6} {bert_metrics['FN']:>6}")
    
    print(f"\n✅ Results saved to: {output_file}")
    print("=" * 80)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python run_all_baselines_csv.py <input_csv> [output_prefix]")
        print("Example: python run_all_baselines_csv.py data/ground_truth_1000_pairs.csv data/1000")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_prefix = sys.argv[2] if len(sys.argv) > 2 else "data/results"
    
    run_all_baselines(input_csv, output_prefix)
