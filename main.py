#!/usr/bin/env python3
"""
Semantic Checker for Open Data - Main Entry Point
==================================================

This is the primary entry point for the Semantic Checker project.
It provides a command-line interface to run different semantic matching baselines
and utilities for dataset generation and analysis.

Usage:
    python main.py --help
    python main.py sample <ground_truth_csv> <output_csv> [--n-positive N] [--n-negative N]
    python main.py run-all <input_csv> [--output <prefix>] [--ollama]
    python main.py baseline <method> <input_csv> <output_csv>
    python main.py check-duplicates <input_csv>
    python main.py dedupe <input_csv> <output_csv>

Author: Aditya Vikram
Date: 3 February 2026
"""

import argparse
import sys
import os
import traceback
from pathlib import Path
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_all_baselines(input_csv: str, output_prefix: str = "results/comparisons/results",
                      run_ollama: bool = False):
    """Run all baselines (Levenshtein, Jaro-Winkler, SBERT, optionally Ollama) on input CSV."""
    from run_all_baselines_csv import run_all_baselines as run_all
    
    if not os.path.exists(input_csv):
        print(f"❌ Error: Input file not found: {input_csv}")
        sys.exit(1)
    
    # Ensure output directory exists
    output_dir = Path(output_prefix).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Running all baselines on: {input_csv}\n")
    run_all(input_csv, output_prefix, run_ollama=run_ollama)


def run_single_baseline(method: str, input_csv: str, output_csv: str):
    """Run a single baseline method on input CSV."""
    if not os.path.exists(input_csv):
        print(f"❌ Error: Input file not found: {input_csv}")
        sys.exit(1)
    
    # Ensure output directory exists
    output_dir = Path(output_csv).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    method = method.lower()
    print(f"🔍 Running {method.upper()} baseline on: {input_csv}\n")
    
    if method == "levenshtein":
        from baselines.levenshtein import analyze_pairs
        analyze_pairs(input_csv, output_csv)
    elif method == "jaro-winkler" or method == "jaro_winkler":
        from baselines.jaro_winkler import analyze_pairs
        analyze_pairs(input_csv, output_csv)
    elif method == "sbert":
        from baselines.sbert import analyze_pairs
        analyze_pairs(input_csv, output_csv)
    elif method in ("ollama", "ollama-zeroshot", "ollama-fewshot", "ollama-cot"):
        from baselines.ollama_prompting import analyze_pairs as ollama_analyze
        # Map shorthand aliases to technique lists
        tech_map = {
            "ollama":          ["zero_shot", "few_shot", "cot"],
            "ollama-zeroshot": ["zero_shot"],
            "ollama-fewshot":  ["few_shot"],
            "ollama-cot":      ["cot"],
        }
        # model can be overridden via OLLAMA_MODEL env var
        import os
        ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
        ollama_analyze(
            data_file=input_csv,
            output_csv=output_csv,
            model=ollama_model,
            techniques=tech_map[method],
        )
    else:
        print(f"❌ Error: Unknown method '{method}'")
        print("Available methods: levenshtein, jaro-winkler, sbert, "
              "ollama, ollama-zeroshot, ollama-fewshot, ollama-cot")
        sys.exit(1)
    
    print(f"\n✅ Results saved to: {output_csv}")


def sample_from_ground_truth(
    ground_truth_csv: str,
    output_csv: str,
    n_positive: int = 10,
    n_negative: int = 10,
    seed: int = 42,
):
    """
    Draw a stratified, balanced test set from a ground-truth CSV and save it.
    This is the recommended way to create test sets instead of manual curation.
    """
    from utils.ground_truth_loader import load_ground_truth
    from utils.stratified_sampler import sample_test_set

    gt = load_ground_truth(ground_truth_csv)
    sample_test_set(gt, n_positive=n_positive, n_negative=n_negative,
                    seed=seed, output_path=output_csv)


def check_duplicates(input_csv: str):
    """Check for duplicate attribute pairs in CSV."""
    if not os.path.exists(input_csv):
        print(f"❌ Error: Input file not found: {input_csv}")
        sys.exit(1)
    
    print(f"🔍 Checking for duplicates in: {input_csv}\n")
    
    df = pd.read_csv(input_csv)
    
    # Check for ordered duplicates
    df['pair_ordered'] = df.apply(
        lambda row: f"{str(row['Attribute1']).lower()}|||{str(row['Attribute2']).lower()}", axis=1
    )
    ordered_dupes = df[df.duplicated(subset=['pair_ordered'], keep=False)]
    
    # Check for unordered duplicates
    df['pair_unordered'] = df.apply(
        lambda row: tuple(sorted([str(row['Attribute1']).lower(), str(row['Attribute2']).lower()])), axis=1
    )
    unordered_dupes = df[df.duplicated(subset=['pair_unordered'], keep=False)]
    
    print(f"📊 Dataset: {len(df)} total pairs")
    print(f"   Ordered duplicates: {len(ordered_dupes) // 2} pairs")
    print(f"   Unordered duplicates: {len(unordered_dupes) // 2} pairs")
    
    if len(ordered_dupes) > 0:
        print(f"\n⚠️  Found {len(ordered_dupes)} rows with ordered duplicate pairs")
    
    if len(unordered_dupes) > len(ordered_dupes):
        print(f"⚠️  Found {len(unordered_dupes)} rows with unordered duplicate pairs")
        print("   (e.g., 'A,B' and 'B,A' treated as same pair)")


def dedupe_csv(input_csv: str, output_csv: str):
    """Remove duplicate attribute pairs from CSV."""
    if not os.path.exists(input_csv):
        print(f"❌ Error: Input file not found: {input_csv}")
        sys.exit(1)
    
    print(f"🧹 Deduplicating: {input_csv}")
    print(f"📝 Output: {output_csv}\n")
    
    df = pd.read_csv(input_csv)
    original_count = len(df)
    
    # Create unordered pair key (case-insensitive)
    df['pair_key'] = df.apply(
        lambda row: tuple(sorted([str(row['Attribute1']).lower(), str(row['Attribute2']).lower()])), axis=1
    )
    
    # Remove duplicates
    df_deduped = df.drop_duplicates(subset=['pair_key'], keep='first')
    df_deduped = df_deduped.drop(columns=['pair_key'])
    
    # Re-sequence IDs
    df_deduped['ID'] = range(1, len(df_deduped) + 1)
    
    # Save
    df_deduped.to_csv(output_csv, index=False)
    
    removed_count = original_count - len(df_deduped)
    print(f"✅ Removed {removed_count} duplicate pairs")
    print(f"📊 Original: {original_count} pairs → Final: {len(df_deduped)} pairs")
    print(f"💾 Saved to: {output_csv}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Semantic Checker for Open Data - Main Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow:
  Step 1 – Create ground truth manually (CSV with Attribute1, Attribute2, Match, Category)
  Step 2 – Sample a balanced test set from it
  Step 3 – Run all baselines on the test set

Examples:
  # STEP 2: Sample a 20-pair balanced test set (10 pos + 10 neg) from a ground truth
  python main.py sample datasets/ground_truth/de_de_ground_truth.csv \\
      datasets/samples/test/test_de_de.csv --n-positive 10 --n-negative 10 --seed 42

  # STEP 3: Run string-similarity baselines only on the sampled test set
  python main.py run-all datasets/samples/test/test_de_de.csv \\
      --output results/experiments/test_run_de_de

  # STEP 3b: Run all baselines including Ollama (requires: ollama serve && ollama pull llama3.1:8b)
  python main.py run-all datasets/samples/test/test_de_de.csv \\
      --output results/experiments/test_run_de_de --ollama

  # Run single baseline
  python main.py baseline sbert datasets/samples/test/test_de_de.csv \\
      results/experiments/sbert/de_de_results.csv

  # Run Ollama few-shot only
  python main.py baseline ollama-fewshot datasets/samples/test/test_de_de.csv \\
      results/experiments/ollama/de_de_fewshot.csv

  # Check for duplicates in ground truth
  python main.py check-duplicates datasets/ground_truth/de_de_ground_truth.csv

  # Remove duplicates from ground truth
  python main.py dedupe datasets/ground_truth/de_de_ground_truth.csv \\
      datasets/ground_truth/de_de_ground_truth_clean.csv
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # run-all command
    parser_run_all = subparsers.add_parser(
        'run-all',
        help='Run all baselines (Levenshtein, Jaro-Winkler, SBERT, optionally Ollama)'
    )
    parser_run_all.add_argument('input_csv', help='Input CSV file with attribute pairs')
    parser_run_all.add_argument('--output', '-o', default='results/comparisons/results',
                                help='Output prefix for results (default: results/comparisons/results)')
    parser_run_all.add_argument('--ollama', action='store_true',
                                help='Also run the Ollama LLM baseline (slow; requires ollama serve)')
    
    # baseline command
    parser_baseline = subparsers.add_parser(
        'baseline',
        help='Run a single baseline method'
    )
    parser_baseline.add_argument('method',
                                choices=['levenshtein', 'jaro-winkler', 'sbert',
                                         'ollama', 'ollama-zeroshot',
                                         'ollama-fewshot', 'ollama-cot'],
                                help='Baseline method to run')
    parser_baseline.add_argument('input_csv', help='Input CSV file with attribute pairs')
    parser_baseline.add_argument('output_csv', help='Output CSV file for results')
    
    # sample command
    parser_sample = subparsers.add_parser(
        'sample',
        help='Draw a stratified test set from a ground-truth CSV'
    )
    parser_sample.add_argument('ground_truth_csv',
                               help='Ground-truth CSV (must have Attribute1, Attribute2, Match columns)')
    parser_sample.add_argument('output_csv',
                               help='Output path for the sampled test set CSV')
    parser_sample.add_argument('--n-positive', '-p', type=int, default=10,
                               help='Number of positive pairs to include (default: 10)')
    parser_sample.add_argument('--n-negative', '-n', type=int, default=10,
                               help='Number of negative pairs to include (default: 10)')
    parser_sample.add_argument('--seed', '-s', type=int, default=42,
                               help='Random seed for reproducibility (default: 42)')

    # check-duplicates command
    parser_check = subparsers.add_parser(
        'check-duplicates',
        help='Check for duplicate attribute pairs in CSV'
    )
    parser_check.add_argument('input_csv', help='Input CSV file to check')
    
    # dedupe command
    parser_dedupe = subparsers.add_parser(
        'dedupe',
        help='Remove duplicate attribute pairs from CSV'
    )
    parser_dedupe.add_argument('input_csv', help='Input CSV file')
    parser_dedupe.add_argument('output_csv', help='Output CSV file (deduplicated)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    try:
        if args.command == 'run-all':
            run_all_baselines(args.input_csv, args.output,
                              run_ollama=getattr(args, 'ollama', False))
        elif args.command == 'sample':
            sample_from_ground_truth(
                args.ground_truth_csv,
                args.output_csv,
                n_positive=args.n_positive,
                n_negative=args.n_negative,
                seed=args.seed,
            )
        elif args.command == 'baseline':
            run_single_baseline(args.method, args.input_csv, args.output_csv)
        elif args.command == 'check-duplicates':
            check_duplicates(args.input_csv)
        elif args.command == 'dedupe':
            dedupe_csv(args.input_csv, args.output_csv)
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
