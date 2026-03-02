#!/usr/bin/env python3
"""
Ollama LLM Semantic Similarity Baseline - Three Prompting Techniques
=====================================================================

Evaluates attribute name equivalence using a local Ollama model with:
  1. Zero-shot  – direct yes/no question, no examples
  2. Few-shot   – 18 hard positive/negative examples before the question
  3. Chain-of-Thought (CoT) – guided step-by-step reasoning

The LLM output is a binary prediction (1=match, 0=no-match) extracted from the
free-text response, so all three techniques are directly comparable to the string
similarity baselines (Levenshtein, Jaro-Winkler, SBERT).

Usage:
    python baselines/ollama_prompting.py --input datasets/ground_truth/open_data_ground_truth.csv \\
                                         --output results/experiments/ollama/results.csv \\
                                         --model llama3.1:8b
    
    # Use a faster/smaller model for development
    python baselines/ollama_prompting.py --input datasets/ground_truth/en_en_ground_truth.csv \\
                                         --model llama3.2:3b

Prerequisites:
    - Ollama installed and running: `ollama serve`
    - Model pulled, e.g.: `ollama pull llama3.1:8b`
    - pip install ollama tqdm

Author: Aditya Vikram
Date: 27 February 2026
"""

import sys
import re
import time
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup – resolved via pyproject.toml when installed with `pip install -e .`
# The insert below is retained as a fallback for running the script directly.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.data_loader import load_pair_data
from utils.evaluation import calculate_metrics

try:
    import ollama as _ollama_lib
except ImportError:
    _ollama_lib = None


# ===========================================================================
# 1.  RESPONSE PARSER
# ===========================================================================

def parse_yes_no(response: str) -> int:
    """Convert a free-text LLM response to a binary prediction.

    Parsing rules (applied in order of precedence):
      1. First word of response is "yes" or "no" (most common in compliant models)
      2. "Final Answer: Yes/No" or "Answer: Yes/No" pattern
      3. "yes" appears but "no" does not (unambiguous positive)
      4. "no" appears but "yes" does not (unambiguous negative)
      5. Default → 0 (conservative: prefer precision over recall on ambiguous output)

    Returns:
        1 if the response indicates a match, 0 otherwise.
    """
    text = response.lower().strip()

    # Rule 1 – leading yes/no
    if text.startswith("yes"):
        return 1
    if text.startswith("no"):
        return 0

    # Rule 2 – explicit labelled answer (CoT and few-shot responses often use this)
    match = re.search(r"(?:final\s+answer|answer)\s*[:\-–]\s*(yes|no)", text)
    if match:
        return 1 if match.group(1) == "yes" else 0

    # Rule 3/4 – only one of yes/no present
    has_yes = bool(re.search(r"\byes\b", text))
    has_no = bool(re.search(r"\bno\b", text))
    if has_yes and not has_no:
        return 1
    if has_no and not has_yes:
        return 0

    # Rule 5 – ambiguous; default to 0
    return 0


# ===========================================================================
# 2.  PROMPT BUILDERS
# ===========================================================================

def zero_shot_prompt(attr1: str, attr2: str) -> str:
    """Zero-shot: single direct question with no examples.

    Strategy:
      - Minimal context to avoid biasing the model.
      - Explicitly mentions cross-language possibility (German/English open-data context).
      - Requests a one-word Yes/No answer for clean parsing.
    """
    return (
        "Determine if the following two data attribute names are semantically equivalent "
        "(i.e., they describe the same type of information). "
        "They may use different naming conventions (camelCase, snake_case, abbreviations) "
        "or be in different languages (e.g., German and English).\n\n"
        f'Attribute 1: "{attr1}"\n'
        f'Attribute 2: "{attr2}"\n\n'
        "Answer with only Yes or No."
    )


def few_shot_prompt(attr1: str, attr2: str) -> str:
    """Few-shot: 18 carefully balanced hard examples followed by the target pair.

    Design principles:
      - Equal split of hard positives (9) and hard negatives (9).
      - Hard positives cover: cross-lingual (DE/EN), naming convention differences,
        abbreviations, synonyms, compound variants.
      - Hard negatives cover: related-but-different concepts (start/end dates,
        billing/shipping address, total/unit price, min/max age etc.).
      - Diverse domain coverage: demographics, addresses, commerce, identifiers.
    """
    examples = """\
HARD POSITIVE – Example 1:
Attribute 1: "BirthDate"
Attribute 2: "Geburtsdatum"
Answer: Yes  (English vs German for date of birth)

HARD POSITIVE – Example 2:
Attribute 1: "CustomerID"
Attribute 2: "Client_Number"
Answer: Yes  (synonym pair; both uniquely identify a customer)

HARD POSITIVE – Example 3:
Attribute 1: "PhoneNumber"
Attribute 2: "Telefonnummer"
Answer: Yes  (English vs German for phone number)

HARD POSITIVE – Example 4:
Attribute 1: "ZipCode"
Attribute 2: "PostalCode"
Answer: Yes  (synonymous terms for the same postal identifier)

HARD POSITIVE – Example 5:
Attribute 1: "DateOfBirth"
Attribute 2: "DOB"
Answer: Yes  (full form and its standard abbreviation)

HARD POSITIVE – Example 6:
Attribute 1: "CompanyName"
Attribute 2: "Firma"
Answer: Yes  (English vs German for company/firm)

HARD POSITIVE – Example 7:
Attribute 1: "AccountNumber"
Attribute 2: "Account_No"
Answer: Yes  (full form vs abbreviated form of the same concept)

HARD POSITIVE – Example 8:
Attribute 1: "LastName"
Attribute 2: "Nachname"
Answer: Yes  (English vs German for surname)

HARD POSITIVE – Example 9:
Attribute 1: "population_count"
Attribute 2: "EinwohnerAnzahl"
Answer: Yes  (snake_case English vs PascalCase German for population count)

HARD NEGATIVE – Example 10:
Attribute 1: "StartDate"
Attribute 2: "EndDate"
Answer: No  (both are dates but mark opposite temporal boundaries)

HARD NEGATIVE – Example 11:
Attribute 1: "StreetAddress"
Attribute 2: "StreetName"
Answer: No  (address includes house number + street; street name is only the road)

HARD NEGATIVE – Example 12:
Attribute 1: "HomeAddress"
Attribute 2: "WorkAddress"
Answer: No  (both are addresses but for different locations)

HARD NEGATIVE – Example 13:
Attribute 1: "TotalPrice"
Attribute 2: "UnitPrice"
Answer: No  (aggregate total vs per-item price)

HARD NEGATIVE – Example 14:
Attribute 1: "EmployeeID"
Attribute 2: "DepartmentID"
Answer: No  (both are identifiers but for different entities)

HARD NEGATIVE – Example 15:
Attribute 1: "CreatedDate"
Attribute 2: "ModifiedDate"
Answer: No  (creation timestamp vs last-modification timestamp)

HARD NEGATIVE – Example 16:
Attribute 1: "MinimumAge"
Attribute 2: "MaximumAge"
Answer: No  (both are age bounds but represent opposite limits)

HARD NEGATIVE – Example 17:
Attribute 1: "BillingAddress"
Attribute 2: "ShippingAddress"
Answer: No  (both are addresses but serve different business purposes)

HARD NEGATIVE – Example 18:
Attribute 1: "PurchaseDate"
Attribute 2: "ReturnDate"
Answer: No  (opposite ends of a transaction lifecycle)

Now determine:
Attribute 1: "{attr1}"
Attribute 2: "{attr2}"

Answer with only Yes or No."""

    return examples.format(attr1=attr1, attr2=attr2)


def chain_of_thought_prompt(attr1: str, attr2: str) -> str:
    """Chain-of-Thought: structured 4-step reasoning before the final decision.
    return (
        "Think step by step to decide whether the two data attribute names below are "
        "semantically equivalent (i.e., they would store the same type of information "
        "in a database).\n\n"
        f'Attribute 1: "{attr1}"\n'
        f'Attribute 2: "{attr2}"\n\n'
        "Step 1 – Attribute 1 meaning: What real-world concept does Attribute 1 represent?\n"
        "Step 2 – Attribute 2 meaning: What real-world concept does Attribute 2 represent?\n"
        "Step 3 – Cross-language check: Could one be a translation or transliteration of the other "
        "(e.g., German ↔ English)?\n"
        "Step 4 – False-friend check: Are they merely related (e.g., start date vs end date, "
        "total price vs unit price) rather than truly equivalent?\n\n"
        "Based on the above reasoning, state your conclusion.\n"
        "Final Answer (Yes or No):"
    )


# ===========================================================================
# 3.  OLLAMA INFERENCE HELPER
# ===========================================================================

def _call_ollama(model: str, prompt: str, timeout: int = 120) -> str:
    """Send a single prompt to the Ollama chat API and return the raw text reply."""
    if _ollama_lib is None:
        raise RuntimeError(
            "The 'ollama' Python package is not installed. "
            "Run: pip install ollama"
        )
    response = _ollama_lib.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1},   # deterministic output
    )
    return response["message"]["content"].strip()


# ===========================================================================
# 4.  MAIN ANALYSIS FUNCTION
# ===========================================================================

def analyze_pairs(
    data_file: str,
    output_csv: str = "results/experiments/ollama/ollama_results.csv",
    model: str = "llama3.1:8b",
    techniques: list = None,
    max_pairs: int = None,
) -> pd.DataFrame:
    """Run all three prompting techniques on every pair and save results.

    Args:
        data_file:  Path to the input ground-truth CSV.
        output_csv: Where to write the results CSV.
        model:      Ollama model tag (must already be pulled via `ollama pull <model>`).
        techniques: Subset of ["zero_shot", "few_shot", "cot"]; defaults to all three.
        max_pairs:  Limit number of rows (useful for quick smoke-tests).

    Returns:
        DataFrame with original columns plus three prediction columns and a
        per-row latency column.
    """
    if techniques is None:
        techniques = ["zero_shot", "few_shot", "cot"]

    print("=" * 80)
    print("🦙  OLLAMA LLM – THREE PROMPTING TECHNIQUES")
    print(f"    Model     : {model}")
    print(f"    Techniques: {', '.join(techniques)}")
    print(f"    Input     : {data_file}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 4a. Check Ollama connectivity
    # ------------------------------------------------------------------
    if _ollama_lib is None:
        print("❌  'ollama' package not installed. Run: pip install ollama")
        sys.exit(1)

    try:
        _ollama_lib.list()
        print("\n✅  Ollama connection OK")
    except Exception as exc:
        print(f"\n❌  Cannot reach Ollama: {exc}")
        print("    Start the server with:  ollama serve")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4b. Load data
    # ------------------------------------------------------------------
    print(f"\nLoading data from: {data_file}")
    try:
        df = load_pair_data(data_file)
    except Exception as exc:
        print(f"❌  Failed to load data: {exc}")
        return pd.DataFrame()

    if max_pairs is not None:
        df = df.head(max_pairs)
        print(f"⚠️   Limited to first {max_pairs} pairs for testing")

    print(f"Total pairs to evaluate: {len(df)}")
    total_llm_calls = len(df) * len(techniques)
    print(f"Total LLM calls planned : {total_llm_calls} "
          f"({len(techniques)} technique(s) × {len(df)} pairs)\n")

    # ------------------------------------------------------------------
    # 4c. Inference loop – one tqdm bar per technique
    # ------------------------------------------------------------------
    prompt_fn_map = {
        "zero_shot": zero_shot_prompt,
        "few_shot":  few_shot_prompt,
        "cot":       chain_of_thought_prompt,
    }

    technique_labels = {
        "zero_shot": "Zero-Shot",
        "few_shot":  "Few-Shot",
        "cot":       "Chain-of-Thought",
    }

    result_cols = {}   # technique_key → list of binary predictions
    latency_cols = {}  # technique_key → list of seconds per call
    raw_response_cols = {}  # technique_key → list of raw text (for debug)

    for tech in techniques:
        preds, latencies, raws = [], [], []
        build_prompt = prompt_fn_map[tech]
        label = technique_labels[tech]

        print(f"\n{'─'*60}")
        print(f"  Running: {label}")
        print(f"{'─'*60}")

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {label:<20}"):
            attr1 = str(row["Attribute1"])
            attr2 = str(row["Attribute2"])
            prompt = build_prompt(attr1, attr2)

            t0 = time.perf_counter()
            try:
                raw = _call_ollama(model, prompt)
                pred = parse_yes_no(raw)
            except Exception as exc:
                tqdm.write(f"  ⚠️  Error on ({attr1!r}, {attr2!r}): {exc}")
                raw, pred = "", 0

            elapsed = time.perf_counter() - t0
            preds.append(pred)
            latencies.append(round(elapsed, 3))
            raws.append(raw)

        result_cols[tech] = preds
        latency_cols[tech] = latencies
        raw_response_cols[tech] = raws

    # ------------------------------------------------------------------
    # 4d. Assemble results DataFrame
    # ------------------------------------------------------------------
    results_df = df.copy()
    for tech in techniques:
        col = f"llama_{tech}"
        results_df[col] = result_cols[tech]
        results_df[f"latency_{tech}_s"] = latency_cols[tech]
        results_df[f"raw_response_{tech}"] = raw_response_cols[tech]

    # ------------------------------------------------------------------
    # 4e. Metrics per technique (only when ground truth is available)
    # ------------------------------------------------------------------
    has_ground_truth = "Match" in results_df.columns or "match" in results_df.columns

    print("\n" + "=" * 80)
    print("📊  RESULTS SUMMARY")
    print("=" * 80)

    summary_rows = []
    for tech in techniques:
        col = f"llama_{tech}"
        label = technique_labels[tech]
        avg_latency = np.mean(latency_cols[tech])
        predicted_matches = int(results_df[col].sum())

        print(f"\n  {label}:")
        print(f"    Predicted matches    : {predicted_matches} / {len(results_df)}")
        print(f"    Predicted non-matches: {len(results_df) - predicted_matches} / {len(results_df)}")
        print(f"    Avg latency / call   : {avg_latency:.2f}s")

        row = {
            "Technique":     label,
            "Predicted_Yes": predicted_matches,
            "Predicted_No":  len(results_df) - predicted_matches,
            "Avg_Latency_s": round(avg_latency, 3),
        }

        if has_ground_truth:
            # calculate_metrics expects 0/1 predictions with threshold=0.5
            m = calculate_metrics(results_df, col, threshold=0.5)
            print(f"    F1-Score  : {m['F1']:.2f}%")
            print(f"    Accuracy  : {m['Accuracy']:.2f}%")
            print(f"    Precision : {m['Precision']:.2f}%")
            print(f"    Recall    : {m['Recall']:.2f}%")
            print(f"    TP:{m['TP']}  FP:{m['FP']}  FN:{m['FN']}  TN:{m['TN']}")
            row.update({
                "F1":       round(m["F1"], 4),
                "Accuracy": round(m["Accuracy"], 4),
                "Precision":round(m["Precision"], 4),
                "Recall":   round(m["Recall"], 4),
                "TP": m["TP"], "FP": m["FP"], "FN": m["FN"], "TN": m["TN"],
            })

        summary_rows.append(row)

    # ------------------------------------------------------------------
    # 4f. Technique comparison table
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("📈  TECHNIQUE COMPARISON")
    print("=" * 80)
    summary_df = pd.DataFrame(summary_rows)
    if has_ground_truth:
        summary_df = summary_df.sort_values("F1", ascending=False)
        print(summary_df[["Technique", "F1", "Accuracy", "Precision", "Recall",
                           "Avg_Latency_s"]].to_string(index=False))
    else:
        print("  (No ground truth — showing prediction counts)")
        print(summary_df[["Technique", "Predicted_Yes", "Predicted_No",
                           "Avg_Latency_s"]].to_string(index=False))

    # ------------------------------------------------------------------
    # 4g. Per-category breakdown (only when both Category and Match exist)
    # ------------------------------------------------------------------
    if has_ground_truth and "Category" in results_df.columns and len(techniques) > 0:
        print("\n" + "=" * 80)
        print("📂  PER-CATEGORY ACCURACY  (best technique per category)")
        print("=" * 80)

        # Find the key for the best technique by F1
        best_key = max(techniques,
                       key=lambda t: calculate_metrics(results_df, f"llama_{t}")["F1"])
        best_col = f"llama_{best_key}"

        for cat, grp in results_df.groupby("Category"):
            correct = (grp[best_col] == grp["Match"].astype(int)).sum()
            acc = correct / len(grp) * 100
            print(f"  {cat:<30}  n={len(grp):>3}  Accuracy={acc:.1f}%")

    # ------------------------------------------------------------------
    # 4h. Challenging pairs analysis
    # ------------------------------------------------------------------
    if "zero_shot" in techniques and "cot" in techniques:
        print("\n" + "=" * 80)
        print("🔍  DISAGREEMENTS: Zero-Shot vs Chain-of-Thought")
        print("=" * 80)
        base_cols = ["Attribute1", "Attribute2", "llama_zero_shot", "llama_cot"]
        extra_cols = [c for c in ["Match", "Category"] if c in results_df.columns]
        disagree = results_df[
            results_df["llama_zero_shot"] != results_df["llama_cot"]
        ][base_cols + extra_cols].head(20)
        if len(disagree) > 0:
            print(disagree.to_string(index=False))
        else:
            print("  Zero-Shot and CoT agree on all pairs.")

    # ------------------------------------------------------------------
    # 4i. Save results
    # ------------------------------------------------------------------
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\n✅  Results saved to: {output_csv}")
    print("=" * 80)

    return results_df


# ===========================================================================
# 5.  COMMAND-LINE INTERFACE
# ===========================================================================

def _parse_args():
    script_dir = Path(__file__).resolve().parent.parent
    default_input  = script_dir / "datasets" / "ground_truth" / "open_data_ground_truth.csv"
    default_output = script_dir / "results"  / "experiments" / "ollama" / "ollama_results.csv"

    parser = argparse.ArgumentParser(
        description="Ollama LLM baseline – zero-shot, few-shot, chain-of-thought",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run with llama3.1:8b on all three ground-truth files
  python baselines/ollama_prompting.py --input datasets/ground_truth/open_data_ground_truth.csv

  # Quick smoke-test (first 20 pairs, small model)
  python baselines/ollama_prompting.py --input datasets/ground_truth/en_en_ground_truth.csv \\
      --model llama3.2:3b --max-pairs 20

  # Run only zero-shot and CoT (skip few-shot)
  python baselines/ollama_prompting.py --input datasets/ground_truth/de_de_ground_truth.csv \\
      --techniques zero_shot cot
        """,
    )
    parser.add_argument(
        "--input", "-i",
        default=str(default_input),
        help="Path to input ground-truth CSV",
    )
    parser.add_argument(
        "--output", "-o",
        default=str(default_output),
        help="Path to output results CSV",
    )
    parser.add_argument(
        "--model", "-m",
        default="llama3.1:8b",
        help="Ollama model tag (default: llama3.1:8b)",
    )
    parser.add_argument(
        "--techniques",
        nargs="+",
        choices=["zero_shot", "few_shot", "cot"],
        default=["zero_shot", "few_shot", "cot"],
        help="Which prompting techniques to run (default: all three)",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        metavar="N",
        help="Limit to first N pairs (useful for quick tests)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    analyze_pairs(
        data_file=args.input,
        output_csv=args.output,
        model=args.model,
        techniques=args.techniques,
        max_pairs=args.max_pairs,
    )
