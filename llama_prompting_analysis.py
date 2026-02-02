#!/usr/bin/env python3
"""
Llama 3.1 Semantic Similarity Analysis with Multiple Prompting Techniques

This script compares three prompting techniques:
1. Zero-shot: Direct question without examples
2. Few-shot: Provide examples before asking
3. Chain-of-Thought (CoT): Step-by-step reasoning

Requires: Ollama running locally with llama3.1 model
"""

import pandas as pd
import ollama
import re
import sys
import argparse
import time
from tqdm import tqdm


def parse_yes_no(response: str) -> int:
    """Parse model response to extract Yes/No answer."""
    response_lower = response.lower().strip()
    
    # Check for explicit yes/no at the start or as standalone
    if response_lower.startswith('yes') or response_lower == 'yes':
        return 1
    if response_lower.startswith('no') or response_lower == 'no':
        return 0
    
    # Check for "Answer: Yes/No" pattern
    match = re.search(r'answer[:\s]*(yes|no)', response_lower)
    if match:
        return 1 if match.group(1) == 'yes' else 0
    
    # Check for yes/no anywhere in response
    if 'yes' in response_lower and 'no' not in response_lower:
        return 1
    if 'no' in response_lower and 'yes' not in response_lower:
        return 0
    
    # Default to no match if unclear
    return 0


def zero_shot_prompt(attr1: str, attr2: str) -> str:
    """Zero-shot prompt: Direct question without examples."""
    return f"""Determine if these two data attribute names are semantically equivalent (mean the same thing).
They may be in different languages (e.g., German and English).

Attribute 1: "{attr1}"
Attribute 2: "{attr2}"

Answer with only Yes or No."""


def few_shot_prompt(attr1: str, attr2: str) -> str:
    """Few-shot prompt with hard positives and negatives."""
    return f"""Determine if two data attribute names are semantically equivalent. Here are challenging examples:

HARD POSITIVE Example 1:
Attribute 1: "BirthDate"
Attribute 2: "Geburtsdatum"
Answer: Yes (both mean date of birth, English and German)

HARD POSITIVE Example 2:
Attribute 1: "CustomerID"
Attribute 2: "Client_Number"
Answer: Yes (both uniquely identify a customer/client)

HARD NEGATIVE Example 3:
Attribute 1: "StartDate"
Attribute 2: "EndDate"
Answer: No (both are dates but represent beginning vs end)

HARD NEGATIVE Example 4:
Attribute 1: "StreetAddress"
Attribute 2: "StreetName"
Answer: No (address includes number and street, street name is only the street)

HARD POSITIVE Example 5:
Attribute 1: "PhoneNumber"
Attribute 2: "Telefonnummer"
Answer: Yes (both mean phone number, English and German)

HARD NEGATIVE Example 6:
Attribute 1: "HomeAddress"
Attribute 2: "WorkAddress"
Answer: No (both are addresses but different locations)

HARD POSITIVE Example 7:
Attribute 1: "ZipCode"
Attribute 2: "PostalCode"
Answer: Yes (synonymous terms for postal code)

HARD NEGATIVE Example 8:
Attribute 1: "EmployeeID"
Attribute 2: "DepartmentID"
Answer: No (both are identifiers but for different entities)

HARD POSITIVE Example 9:
Attribute 1: "CompanyName"
Attribute 2: "Firma"
Answer: Yes (both mean company/firm, English and German)

HARD NEGATIVE Example 10:
Attribute 1: "CreatedDate"
Attribute 2: "ModifiedDate"
Answer: No (both are timestamps but represent different events)

HARD POSITIVE Example 11:
Attribute 1: "EmailAddress"
Attribute 2: "E_Mail_Adresse"
Answer: Yes (both mean email address, English and German)

HARD NEGATIVE Example 12:
Attribute 1: "TotalPrice"
Attribute 2: "UnitPrice"
Answer: No (both are prices but total vs per-unit)

HARD POSITIVE Example 13:
Attribute 1: "LastName"
Attribute 2: "Nachname"
Answer: Yes (both mean last name/surname, English and German)

HARD NEGATIVE Example 14:
Attribute 1: "BillingAddress"
Attribute 2: "ShippingAddress"
Answer: No (both are addresses but for different purposes)

HARD POSITIVE Example 15:
Attribute 1: "DateOfBirth"
Attribute 2: "DOB"
Answer: Yes (full form and abbreviation of same concept)

HARD NEGATIVE Example 16:
Attribute 1: "MinimumAge"
Attribute 2: "MaximumAge"
Answer: No (both are age limits but represent opposite bounds)

HARD POSITIVE Example 17:
Attribute 1: "AccountNumber"
Attribute 2: "Account_No"
Answer: Yes (full form and abbreviated form of same concept)

HARD NEGATIVE Example 18:
Attribute 1: "PurchaseDate"
Attribute 2: "ReturnDate"
Answer: No (both are transaction dates but opposite operations)

Now determine:
Attribute 1: "{attr1}"
Attribute 2: "{attr2}"

Answer with only Yes or No."""


def chain_of_thought_prompt(attr1: str, attr2: str) -> str:
    """Chain-of-Thought prompt: Step-by-step reasoning."""
    return f"""Think step by step to determine if these data attribute names are semantically equivalent:

Attribute 1: "{attr1}"
Attribute 2: "{attr2}"

Steps:
1. What does Attribute 1 likely represent in a database context?
2. What does Attribute 2 likely represent in a database context?
3. Are they describing the same type of information?
4. Consider: Could they be the same concept in different languages?

Based on your reasoning, answer: Are they semantically equivalent?
Final Answer (Yes/No):"""


def run_llama_analysis(input_csv: str, output_csv: str, model: str = "llama3.1:8b"):
    """Run Llama 3.1 analysis with all three prompting techniques."""
    
    print("=" * 80)
    print(f"🦙 LLAMA 3.1 PROMPTING TECHNIQUES ANALYSIS")
    print(f"   Model: {model}")
    print(f"   Input: {input_csv}")
    print("=" * 80)
    
    # Load data
    df = pd.read_csv(input_csv)
    print(f"\n✅ Loaded {len(df)} pairs")
    
    # Check if Ollama is available
    try:
        ollama.list()
        print("✅ Ollama connection successful")
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        print("   Make sure Ollama is running: ollama serve")
        sys.exit(1)
    
    # Initialize result columns
    zeroshot_scores = []
    fewshot_scores = []
    cot_scores = []
    
    # Process each pair
    print("\n📊 Processing pairs with 3 prompting techniques...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        attr1 = str(row['Attribute1'])
        attr2 = str(row['Attribute2'])
        
        # 1. Zero-shot
        try:
            response = ollama.chat(model=model, messages=[
                {'role': 'user', 'content': zero_shot_prompt(attr1, attr2)}
            ])
            zeroshot_scores.append(parse_yes_no(response['message']['content']))
        except Exception as e:
            print(f"\n⚠️ Zero-shot error at row {idx}: {e}")
            zeroshot_scores.append(0)
        
        # 2. Few-shot
        try:
            response = ollama.chat(model=model, messages=[
                {'role': 'user', 'content': few_shot_prompt(attr1, attr2)}
            ])
            fewshot_scores.append(parse_yes_no(response['message']['content']))
        except Exception as e:
            print(f"\n⚠️ Few-shot error at row {idx}: {e}")
            fewshot_scores.append(0)
        
        # 3. Chain-of-Thought
        try:
            response = ollama.chat(model=model, messages=[
                {'role': 'user', 'content': chain_of_thought_prompt(attr1, attr2)}
            ])
            cot_scores.append(parse_yes_no(response['message']['content']))
        except Exception as e:
            print(f"\n⚠️ CoT error at row {idx}: {e}")
            cot_scores.append(0)
        
        # Progress update every 100 rows
        if (idx + 1) % 100 == 0:
            print(f"\n  Processed {idx + 1}/{len(df)} pairs...")
    
    # Add results to dataframe
    df['llama_zeroshot'] = zeroshot_scores
    df['llama_fewshot'] = fewshot_scores
    df['llama_cot'] = cot_scores
    
    # Calculate metrics for each technique
    print("\n" + "=" * 80)
    print("📊 RESULTS SUMMARY")
    print("=" * 80)
    
    techniques = [
        ('Zero-shot', 'llama_zeroshot'),
        ('Few-shot', 'llama_fewshot'),
        ('Chain-of-Thought', 'llama_cot')
    ]
    
    results = []
    for name, col in techniques:
        predictions = df[col]
        actuals = df['Match'].astype(int)
        
        tp = ((predictions == 1) & (actuals == 1)).sum()
        fp = ((predictions == 1) & (actuals == 0)).sum()
        fn = ((predictions == 0) & (actuals == 1)).sum()
        tn = ((predictions == 0) & (actuals == 0)).sum()
        
        accuracy = (tp + tn) / len(df) if len(df) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            'Technique': name,
            'F1': f1,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'TN': tn
        })
        
        print(f"\n{name}:")
        print(f"  F1-Score:  {f1*100:.2f}%")
        print(f"  Accuracy:  {accuracy*100:.2f}%")
        print(f"  Precision: {precision*100:.2f}%")
        print(f"  Recall:    {recall*100:.2f}%")
        print(f"  TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    
    # Save results
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Results saved to: {output_csv}")
    
    # Print comparison
    print("\n" + "=" * 80)
    print("📈 TECHNIQUE COMPARISON (sorted by F1-Score)")
    print("=" * 80)
    results_df = pd.DataFrame(results).sort_values('F1', ascending=False)
    print(results_df[['Technique', 'F1', 'Accuracy', 'Precision', 'Recall']].to_string(index=False))
    
    return df


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "data" / "clean_ground_truth_1000.csv"
    default_output = script_dir / "data" / "llama_prompting_results.csv"

    parser = argparse.ArgumentParser(description='Run Llama prompting analysis')
    parser.add_argument('input_arg', nargs='?', default=None, help='Legacy positional input argument')
    parser.add_argument('--input', type=str, default=str(default_input), help='Path to input CSV')
    parser.add_argument('--output', type=str, default=str(default_output), help='Path to output CSV')
    parser.add_argument('--model', type=str, default="llama3.1:8b", help='Ollama model name')
    args = parser.parse_args()
    
    # Handle legacy positional arg
    input_csv = args.input_arg if args.input_arg else args.input

    input_path = Path(input_csv)
    if not input_path.exists():
         # Try finding it in data/ relative to CWD or script
         if (Path("data") / input_csv).exists():
             input_path = Path("data") / input_csv
         elif (script_dir / "data" / input_csv).exists():
             input_path = script_dir / "data" / input_csv
         else:
             print(f"Error: {input_csv} not found!")
             sys.exit(1)
    
    run_llama_analysis(str(input_path), args.output, args.model)
