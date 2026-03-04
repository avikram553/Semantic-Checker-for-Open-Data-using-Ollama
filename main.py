#!/usr/bin/env python3
"""
Semantic Checker for Open Data - Main Entry Point
==================================================

This is the primary entry point for the Semantic Checker project.
It provides a command-line interface to run different semantic matching baselines
and utilities for dataset generation and analysis.

Usage:
    python main.py --help
    python main.py run-all <input_csv> [--output <prefix>]
    python main.py baseline <method> <input_csv> <output_csv>
    python main.py generate-dataset <output_csv> [--count <n>]
    python main.py check-duplicates <input_csv>
    python main.py dedupe <input_csv> <output_csv>

Author: Aditya Vikram
Date: 3 February 2026
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_all_baselines(input_csv: str, output_prefix: str = "results/comparisons/results"):
    """Run all baselines (Levenshtein, Jaro-Winkler, SBERT) on input CSV."""
    from run_all_baselines_csv import run_all_baselines as run_all
    
    if not os.path.exists(input_csv):
        print(f"❌ Error: Input file not found: {input_csv}")
        sys.exit(1)
    
    # Ensure output directory exists
    output_dir = Path(output_prefix).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Running all baselines on: {input_csv}\n")
    run_all(input_csv, output_prefix)


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
        tech_map = {
            "ollama":          ["zero_shot", "few_shot", "cot"],
            "ollama-zeroshot": ["zero_shot"],
            "ollama-fewshot":  ["few_shot"],
            "ollama-cot":      ["cot"],
        }
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


def generate_dataset(output_csv: str, count: int = 600):
    """Generate a new ground truth dataset."""
    from utils.generate_open_data_ground_truth import main as generate_main
    
    print(f"🎲 Generating dataset with {count} attribute pairs...")
    print(f"📝 Output: {output_csv}\n")
    
    # Call the generator with output path
    # Note: The generate_open_data_ground_truth.py script needs to be adapted
    # For now, we'll provide instructions
    print("⚠️  To generate dataset, run:")
    print(f"   python utils/generate_open_data_ground_truth.py")
    print(f"   Then move the generated file to: {output_csv}")


def show_ollama_results(results_csv: str, technique: str = "all"):
    """
    Display existing Ollama results from a CSV file in the CLI.

    Args:
        results_csv: Path to an Ollama results CSV file.
        technique:   Which technique(s) to show – 'zero_shot', 'few_shot',
                     'cot', or 'all' (default).
    """
    import pandas as pd
    from utils.evaluation import calculate_metrics

    if not os.path.exists(results_csv):
        print(f"❌ Error: Results file not found: {results_csv}")
        sys.exit(1)

    df = pd.read_csv(results_csv)

    # Discover which technique columns exist
    technique_cols = {
        "zero_shot": "llama_zero_shot",
        "few_shot":  "llama_few_shot",
        "cot":       "llama_cot",
    }
    if technique == "all":
        show_cols = {k: v for k, v in technique_cols.items() if v in df.columns}
    else:
        if technique_cols.get(technique) not in df.columns:
            print(f"❌ Column '{technique_cols.get(technique)}' not found in {results_csv}")
            print(f"   Available columns: {df.columns.tolist()}")
            sys.exit(1)
        show_cols = {technique: technique_cols[technique]}

    print("\n" + "=" * 80)
    print(f"📊  OLLAMA LLM RESULTS  —  {Path(results_csv).name}")
    print(f"    Total pairs: {len(df)}")
    n_pos = int(df["Match"].sum()) if "Match" in df.columns else "?"
    print(f"    Positives: {n_pos}   Negatives: {len(df) - n_pos if isinstance(n_pos, int) else '?'}")
    print("=" * 80)

    for tech_key, col in show_cols.items():
        label = tech_key.replace("_", "-").title()
        print(f"\n{'─' * 80}")
        print(f"  🤖  {label}")
        print(f"{'─' * 80}")

        metrics = calculate_metrics(df, col, threshold=0.5)
        print(f"\n  {'Metric':<12} {'Value':>10}")
        print(f"  {'-' * 24}")
        for key in ("Accuracy", "Precision", "Recall", "F1"):
            print(f"  {key:<12} {metrics[key]:>9.2f}%")
        print(f"  {'TP':<12} {metrics['TP']:>10}")
        print(f"  {'FP':<12} {metrics['FP']:>10}")
        print(f"  {'TN':<12} {metrics['TN']:>10}")
        print(f"  {'FN':<12} {metrics['FN']:>10}")

        lat_col = f"latency_{tech_key}_s"
        if lat_col in df.columns:
            print(f"\n  ⏱  Avg latency : {df[lat_col].mean():.2f}s")
            print(f"     Total time  : {df[lat_col].sum():.1f}s")

        print(f"\n  {'#':<4} {'Attribute1':<22} {'Attribute2':<22} {'GT':>4} {'Pred':>6} {'OK?':>5}")
        print(f"  {'-' * 64}")
        for _, row in df.iterrows():
            gt   = "✅" if row.get("Match") else "❌"
            pred = "✅" if row[col] == 1 else "❌"
            ok   = "✔" if row.get("Match") == (row[col] == 1) else "✘"
            a1 = str(row.get("Attribute1", ""))[:20]
            a2 = str(row.get("Attribute2", ""))[:20]
            print(f"  {int(row.get('ID', 0)):<4} {a1:<22} {a2:<22} {gt:>4} {pred:>6} {ok:>5}")

        if "Category" in df.columns:
            print(f"\n  Category breakdown:")
            for cat, grp in df.groupby("Category"):
                m = calculate_metrics(grp, col, threshold=0.5)
                print(f"    {cat:<28} F1={m['F1']:>5.1f}%  (n={len(grp)})")

    print("\n" + "=" * 80)
    print("  Source file:", results_csv)
    print("=" * 80 + "\n")


def show_all_results(results_csv: str, threshold: float = 0.5):
    """
    Display results for all 6 methods from a combined results CSV.

    Uses method-specific optimal thresholds:
        - Levenshtein  : 0.75
        - Jaro-Winkler : 0.85
        - SBERT        : 0.75
        - LLM methods  : 0.5  (binary 0/1 output)

    Args:
        results_csv: Path to the combined results CSV file.
        threshold:   Fallback threshold for any unrecognised column (default 0.5).
    """
    import pandas as pd
    from utils.evaluation import calculate_metrics

    if not os.path.exists(results_csv):
        print(f"❌ Error: Results file not found: {results_csv}")
        sys.exit(1)

    df = pd.read_csv(results_csv)

    # Map human-readable labels → (column name, threshold)
    METHOD_THRESHOLDS = {
        "Levenshtein":    ("levenshtein_similarity",  0.75),
        "Jaro-Winkler":   ("jaro_winkler_similarity", 0.85),
        "SBERT":          ("sbert_similarity",         0.75),
        "LLM Zero-Shot":  ("llama_zero_shot",          0.50),
        "LLM Few-Shot":   ("llama_few_shot",           0.50),
        "LLM CoT":        ("llama_cot",                0.50),
    }

    # Only show methods whose column is actually present
    available = {
        lbl: (col, thr)
        for lbl, (col, thr) in METHOD_THRESHOLDS.items()
        if col in df.columns
    }
    if not available:
        print(f"❌ No recognised method columns found in {results_csv}")
        print(f"   Columns present: {df.columns.tolist()}")
        sys.exit(1)

    n_total = len(df)
    n_pos   = int(df["Match"].sum()) if "Match" in df.columns else "?"
    n_neg   = (n_total - n_pos) if isinstance(n_pos, int) else "?"

    print("\n" + "=" * 80)
    print(f"📊  ALL-METHODS RESULTS  —  {Path(results_csv).name}")
    print(f"    Pairs: {n_total}  |  Positives: {n_pos}  |  Negatives: {n_neg}")
    print(f"    Thresholds → LEV: 0.75  |  JW: 0.85  |  SBERT: 0.75  |  LLM: 0.50")
    print("=" * 80)

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n  {'Method':<20} {'Thr':>5} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}  {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4}")
    print(f"  {'-' * 76}")
    metrics_cache = {}
    for label, (col, thr) in available.items():
        m = calculate_metrics(df, col, threshold=thr)
        metrics_cache[col] = (m, thr)
        print(
            f"  {label:<20} {thr:>5.2f} {m['Accuracy']:>6.1f}% {m['Precision']:>6.1f}% "
            f"{m['Recall']:>6.1f}% {m['F1']:>6.1f}%  "
            f"{m['TP']:>4} {m['FP']:>4} {m['TN']:>4} {m['FN']:>4}"
        )

    # ── Per-pair detail ────────────────────────────────────────────────────────
    short_headers = {
        "levenshtein_similarity":  "LEV",
        "jaro_winkler_similarity": "JW ",
        "sbert_similarity":        "SBT",
        "llama_zero_shot":         "ZS ",
        "llama_few_shot":          "FS ",
        "llama_cot":               "COT",
    }

    cols_present = [col for col, _ in available.values()]
    hdrs = "  ".join(f"{short_headers[c]:>5}" for c in cols_present)

    print(f"\n  {'#':<4} {'Attribute1':<22} {'Attribute2':<22} {'GT':>3}  {hdrs}")
    print(f"  {'-' * (56 + 7 * len(cols_present))}")

    for _, row in df.iterrows():
        gt = "✅" if row.get("Match") else "❌"
        preds = "  ".join(
            f"{'✔' if row[c] >= thr else '✘':>5}"
            for c, thr in [(col, thr) for col, thr in
                           [(col, metrics_cache[col][1]) for col in cols_present]]
        )
        a1 = str(row.get("Attribute1", ""))[:20]
        a2 = str(row.get("Attribute2", ""))[:20]
        print(f"  {int(row.get('ID', 0)):<4} {a1:<22} {a2:<22} {gt:>3}  {preds}")

    # ── Category breakdown ─────────────────────────────────────────────────────
    if "Category" in df.columns:
        print(f"\n  Category breakdown  (F1 per method)")
        cat_hdr = "  ".join(f"{short_headers[c]:>6}" for c in cols_present)
        print(f"  {'Category':<32}  {cat_hdr}")
        print(f"  {'-' * (34 + 8 * len(cols_present))}")
        for cat, grp in df.groupby("Category"):
            f1s = "  ".join(
                f"{calculate_metrics(grp, c, threshold=metrics_cache[c][1])['F1']:>5.1f}%"
                for c in cols_present
            )
            print(f"  {cat:<32}  {f1s}  (n={len(grp)})")

    print("\n" + "=" * 80)
    print("  Source file :", results_csv)
    print(f"  Thresholds  : LEV=0.75  JW=0.85  SBERT=0.75  LLM=0.50")
    print("=" * 80 + "\n")


def check_duplicates(input_csv: str):
    """Check for duplicate attribute pairs in CSV."""
    if not os.path.exists(input_csv):
        print(f"❌ Error: Input file not found: {input_csv}")
        sys.exit(1)
    
    print(f"🔍 Checking for duplicates in: {input_csv}\n")
    
    import pandas as pd
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
    
    import pandas as pd
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
Examples:
  # Run all baselines on a dataset
  python main.py run-all datasets/ground_truth/open_data_ground_truth.csv --output results/comparisons/experiment1
  
  # Run single baseline
  python main.py baseline sbert datasets/ground_truth/open_data_ground_truth.csv results/experiments/sbert/results.csv
  python main.py baseline levenshtein datasets/ground_truth/open_data_ground_truth.csv results/experiments/levenshtein/results.csv
  
  # Generate new ground truth dataset
  python main.py generate-dataset datasets/ground_truth/new_dataset.csv --count 600
  
  # Check for duplicates
  python main.py check-duplicates datasets/ground_truth/open_data_ground_truth.csv
  
  # Remove duplicates
  python main.py dedupe datasets/ground_truth/input.csv datasets/ground_truth/output_clean.csv
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # run-all command
    parser_run_all = subparsers.add_parser(
        'run-all',
        help='Run all baselines (Levenshtein, Jaro-Winkler, SBERT)'
    )
    parser_run_all.add_argument('input_csv', help='Input CSV file with attribute pairs')
    parser_run_all.add_argument('--output', '-o', default='results/comparisons/results',
                                help='Output prefix for results (default: results/comparisons/results)')
    
    # baseline command
    parser_baseline = subparsers.add_parser(
        'baseline',
        help='Run a single baseline method'
    )
    parser_baseline.add_argument('method',
                                choices=['levenshtein', 'jaro-winkler', 'sbert',
                                         'ollama', 'ollama-zeroshot', 'ollama-fewshot', 'ollama-cot'],
                                help='Baseline method to run')
    parser_baseline.add_argument('input_csv', help='Input CSV file with attribute pairs')
    parser_baseline.add_argument('output_csv', help='Output CSV file for results')
    
    # generate-dataset command
    parser_generate = subparsers.add_parser(
        'generate-dataset',
        help='Generate a new ground truth dataset'
    )
    parser_generate.add_argument('output_csv', help='Output CSV file path')
    parser_generate.add_argument('--count', '-n', type=int, default=600,
                                help='Number of attribute pairs to generate (default: 600)')
    
    # show-results command
    parser_show = subparsers.add_parser(
        'show-results',
        help='Display existing Ollama results from a CSV file in the CLI'
    )
    parser_show.add_argument('results_csv', help='Path to the Ollama results CSV file')
    parser_show.add_argument('--technique', '-t', default='all',
                             choices=['all', 'zero_shot', 'few_shot', 'cot'],
                             help='Technique to display (default: all)')

    # show-all-results command
    parser_show_all = subparsers.add_parser(
        'show-all-results',
        help='Display results for all 6 methods from a combined results CSV'
    )
    parser_show_all.add_argument('results_csv', help='Path to the combined results CSV file')
    parser_show_all.add_argument('--threshold', '-t', type=float, default=0.5,
                                 help='Score threshold for classifying positives (default: 0.5)')

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
            run_all_baselines(args.input_csv, args.output)
        elif args.command == 'baseline':
            run_single_baseline(args.method, args.input_csv, args.output_csv)
        elif args.command == 'show-results':
            show_ollama_results(args.results_csv, args.technique)
        elif args.command == 'show-all-results':
            show_all_results(args.results_csv, args.threshold)
        elif args.command == 'generate-dataset':
            generate_dataset(args.output_csv, args.count)
        elif args.command == 'check-duplicates':
            check_duplicates(args.input_csv)
        elif args.command == 'dedupe':
            dedupe_csv(args.input_csv, args.output_csv)
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
