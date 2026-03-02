# Datasets Directory

This directory contains all datasets used in the semantic checker project.

## Directory Structure

```
datasets/
├── ground_truth/        # Manually curated ground truth datasets
├── generated/           # Automatically generated test sets
└── samples/             # Small sample datasets for testing
```

## Ground Truth Datasets

### Format Requirements
All ground truth CSVs must have these columns:
- `ID`: Unique identifier for the pair
- `Attribute1`: First attribute name
- `Attribute2`: Second attribute name
- `Match`: Boolean indicating if attributes are semantically equivalent
- `Category`: Category of the test case (e.g., "easy_positive", "hard_negative")
- `Confidence`: Confidence level of the label (high/medium/low)
- `Type`: Type of relationship (positive/negative)
- `Reasoning`: Explanation for the label

### Current Datasets

1. **open_data_ground_truth.csv** (600 pairs)
   - Balanced German open data attributes
   - Categories: easy_positive, cross_lingual, paraphrases, abbreviations, hard_negative, toughest_negative
   - Date: 2026-02-03

2. **open_data_1500_master_ground_truth.csv** (1500 pairs)
   - Extended version with more test cases
   - Date: 2026-02-03

3. **hard_negative_sets.csv**
   - Challenging negative examples
   - Focus on similar but non-matching attributes

## File Naming Convention

Format: `{domain}_{type}_{version}.csv`
- Example: `open_data_ground_truth_v2.csv`

## Quality Guidelines

- No duplicate pairs (checked with utils/check_duplicate_pairs.py)
- Balanced distribution of positive/negative examples
- Clear reasoning for each label
- Representative of real-world attribute matching scenarios
