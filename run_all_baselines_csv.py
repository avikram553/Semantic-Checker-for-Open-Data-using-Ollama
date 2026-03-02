import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.string_matching import levenshtein_similarity, jaro_winkler_similarity
from utils.evaluation import calculate_metrics
from utils.data_loader import load_pair_data
from baselines.sbert import load_sbert_model, calculate_sbert_similarity

def run_all_baselines(input_csv, output_prefix, run_ollama=False,
                      ollama_model="llama3.1:8b"):
    print("=" * 80)
    print(f"🚀 RUNNING ALL 3 BASELINES ON: {input_csv}")
    print("=" * 80)
    
    df = load_pair_data(input_csv)
    has_ground_truth = "Match" in df.columns or "match" in df.columns
    print(f"\n✅ Loaded {len(df)} pairs")
    if has_ground_truth:
        match_col = df["Match"] if "Match" in df.columns else df["match"]
        print(f"   Matches: {match_col.sum()}")
        print(f"   Non-matches: {(~match_col).sum()}")
    else:
        print("   ⚠️  No 'Match' column — similarity scores only (no F1/Accuracy)")
    
    results_df = df.copy()
    
    print("\n" + "=" * 80)
    print("1️⃣ LEVENSHTEIN DISTANCE")
    print("=" * 80)
    results_df['levenshtein_similarity'] = results_df.apply(
        lambda row: levenshtein_similarity(str(row['Attribute1']), str(row['Attribute2'])), axis=1
    )
    if has_ground_truth:
        lev_metrics = calculate_metrics(results_df, 'levenshtein_similarity', threshold=0.70)
        print(f"F1-Score: {lev_metrics['F1']:.2f}%")
        print(f"Accuracy: {lev_metrics['Accuracy']:.2f}%")
        print(f"False Positives: {lev_metrics['FP']}, False Negatives: {lev_metrics['FN']}")
    else:
        lev_metrics = {}
        predicted = (results_df['levenshtein_similarity'] >= 0.70).sum()
        print(f"Predicted matches (threshold=0.70): {predicted} / {len(results_df)}")
        print(f"Avg similarity: {results_df['levenshtein_similarity'].mean():.4f}")

    print("\n" + "=" * 80)
    print("2️⃣ JARO-WINKLER DISTANCE")
    print("=" * 80)
    results_df['jaro_winkler_similarity'] = results_df.apply(
        lambda row: jaro_winkler_similarity(str(row['Attribute1']), str(row['Attribute2'])), axis=1
    )
    if has_ground_truth:
        jw_metrics = calculate_metrics(results_df, 'jaro_winkler_similarity', threshold=0.85)
        print(f"F1-Score: {jw_metrics['F1']:.2f}%")
        print(f"Accuracy: {jw_metrics['Accuracy']:.2f}%")
        print(f"False Positives: {jw_metrics['FP']}, False Negatives: {jw_metrics['FN']}")
    else:
        jw_metrics = {}
        predicted = (results_df['jaro_winkler_similarity'] >= 0.85).sum()
        print(f"Predicted matches (threshold=0.85): {predicted} / {len(results_df)}")
        print(f"Avg similarity: {results_df['jaro_winkler_similarity'].mean():.4f}")

    print("\n" + "=" * 80)
    print("3️⃣ SBERT MULTILINGUAL")
    print("=" * 80)
    sbert_model = load_sbert_model()

    results_df['sbert_similarity'] = results_df.apply(
        lambda row: calculate_sbert_similarity(
            str(row['Attribute1']), str(row['Attribute2']), sbert_model
        ),
        axis=1,
    )
    if has_ground_truth:
        sbert_metrics = calculate_metrics(results_df, 'sbert_similarity', threshold=0.75)
        print(f"F1-Score: {sbert_metrics['F1']:.2f}%")
        print(f"Accuracy: {sbert_metrics['Accuracy']:.2f}%")
        print(f"False Positives: {sbert_metrics['FP']}, False Negatives: {sbert_metrics['FN']}")
    else:
        sbert_metrics = {}
        predicted = (results_df['sbert_similarity'] >= 0.75).sum()
        print(f"Predicted matches (threshold=0.75): {predicted} / {len(results_df)}")
        print(f"Avg similarity: {results_df['sbert_similarity'].mean():.4f}")

    # ---------------------------------------------------------------
    # 4. OLLAMA – three prompting techniques (optional, slow)
    # ---------------------------------------------------------------
    ollama_metrics = {}
    if run_ollama:
        print("\n" + "=" * 80)
        print("4️⃣  OLLAMA LLM – THREE PROMPTING TECHNIQUES")
        print(f"    Model: {ollama_model}")
        print("=" * 80)
        try:
            from baselines.ollama_prompting import analyze_pairs as ollama_analyze
            ollama_output = f"{output_prefix}_ollama_results.csv"
            ollama_df = ollama_analyze(
                data_file=input_csv,
                output_csv=ollama_output,
                model=ollama_model,
                techniques=["zero_shot", "few_shot", "cot"],
            )
            for tech, label in [("zero_shot", "Ollama Zero-Shot"),
                                 ("few_shot",  "Ollama Few-Shot"),
                                 ("cot",       "Ollama CoT")]:
                col = f"llama_{tech}"
                if col in ollama_df.columns:
                    m = calculate_metrics(ollama_df, col, threshold=0.5)
                    ollama_metrics[label] = m
                    results_df[col] = ollama_df[col].values
                    print(f"  {label}: F1={m['F1']:.2f}%  Acc={m['Accuracy']:.2f}%")
        except Exception as exc:
            print(f"⚠️  Ollama run failed: {exc}")
            print("   Skipping Ollama section. String-similarity results still saved.")
    else:
        print("\nℹ️   Ollama LLM baseline skipped (pass --ollama to enable).")

    # ---------------------------------------------------------------
    # Summary comparison table
    # ---------------------------------------------------------------
    print("\n" + "=" * 80)
    print("📊  COMPARISON SUMMARY")
    if has_ground_truth:
        print(f"{'Baseline':<30} {'F1':>10} {'Accuracy':>10} {'FP':>6} {'FN':>6}")
        print("-" * 64)
        print(f"{'Levenshtein':<30} {lev_metrics['F1']:>10.2f}% {lev_metrics['Accuracy']:>10.2f}% {lev_metrics['FP']:>6} {lev_metrics['FN']:>6}")
        print(f"{'Jaro-Winkler':<30} {jw_metrics['F1']:>10.2f}% {jw_metrics['Accuracy']:>10.2f}% {jw_metrics['FP']:>6} {jw_metrics['FN']:>6}")
        print(f"{'SBERT Multilingual':<30} {sbert_metrics['F1']:>10.2f}% {sbert_metrics['Accuracy']:>10.2f}% {sbert_metrics['FP']:>6} {sbert_metrics['FN']:>6}")
        for label, m in ollama_metrics.items():
            print(f"{label:<30} {m['F1']:>10.2f}% {m['Accuracy']:>10.2f}% {m['FP']:>6} {m['FN']:>6}")
    else:
        print(f"{'Baseline':<30} {'Predicted YES':>14} {'Avg Sim':>10}")
        print("-" * 58)
        for col, name, thr in [('levenshtein_similarity',  'Levenshtein',        0.70),
                                ('jaro_winkler_similarity', 'Jaro-Winkler',        0.85),
                                ('sbert_similarity',        'SBERT Multilingual',  0.75)]:
            yes = int((results_df[col] >= thr).sum())
            avg = results_df[col].mean()
            print(f"{name:<30} {yes:>14} {avg:>10.4f}")
        for label, m in ollama_metrics.items():
            print(f"  {label}")  # ollama metrics only exist with ground truth

    output_file = f"{output_prefix}_all_baselines_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to: {output_file}")
    print("=" * 80)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run all baselines on a ground-truth CSV"
    )
    parser.add_argument("input_csv", help="Input CSV file with attribute pairs")
    parser.add_argument("output_prefix", nargs="?", default="results/comparisons/results",
                        help="Output file prefix (default: results/comparisons/results)")
    parser.add_argument("--ollama", action="store_true",
                        help="Also run the Ollama LLM baseline (slow)")
    parser.add_argument("--ollama-model", default="llama3.1:8b",
                        help="Ollama model tag (default: llama3.1:8b)")
    args = parser.parse_args()

    run_all_baselines(args.input_csv, args.output_prefix,
                      run_ollama=args.ollama, ollama_model=args.ollama_model)
