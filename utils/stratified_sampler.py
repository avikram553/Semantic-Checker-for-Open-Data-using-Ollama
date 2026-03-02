"""
utils/stratified_sampler.py
----------------------------
Stratified test-set sampler.

Takes a ground-truth DataFrame (loaded via ground_truth_loader) and draws a
balanced test set with a fixed number of positive and negative pairs, spread
proportionally across difficulty strata (Category column).

Usage:
    from utils.ground_truth_loader import load_ground_truth
    from utils.stratified_sampler import sample_test_set

    gt  = load_ground_truth("datasets/ground_truth/de_de_ground_truth.csv")
    test = sample_test_set(gt, n_positive=10, n_negative=10, seed=42)
    test.to_csv("datasets/samples/test/test_de_de.csv", index=False)
"""

import math
from typing import Optional

import pandas as pd
from pathlib import Path


def _stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """
    Draw n rows from df, distributed proportionally across the 'Category'
    column.  Falls back to a plain random sample when n >= len(df) or when
    every stratum would get 0 rows.

    Args:
        df:   DataFrame subset (positives-only OR negatives-only).
        n:    Total number of rows to draw.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with exactly min(n, len(df)) rows.
    """
    if n >= len(df):
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)

    strata = df["Category"].unique()
    per_stratum = max(1, math.floor(n / len(strata)))
    sampled_parts = []

    for stratum in strata:
        subset = df[df["Category"] == stratum]
        take = min(per_stratum, len(subset))
        sampled_parts.append(subset.sample(n=take, random_state=seed))

    sampled = pd.concat(sampled_parts, ignore_index=True)

    # Top-up or trim to exactly n rows
    if len(sampled) < n:
        already_picked = sampled.index
        remaining = df.drop(index=df.index.intersection(sampled.index))
        top_up = remaining.sample(
            n=min(n - len(sampled), len(remaining)), random_state=seed
        )
        sampled = pd.concat([sampled, top_up], ignore_index=True)

    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=seed).reset_index(drop=True)

    return sampled.reset_index(drop=True)


def sample_test_set(
    gt: pd.DataFrame,
    n_positive: int = 10,
    n_negative: int = 10,
    seed: int = 42,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Draw a balanced, stratified test set from a ground-truth DataFrame.

    Args:
        gt:          Full ground-truth DataFrame (from load_ground_truth).
        n_positive:  Number of positive (Match=True) pairs to include.
        n_negative:  Number of negative (Match=False) pairs to include.
        seed:        Random seed — fix this to guarantee reproducibility (NFR3).
        output_path: If provided, saves the test set to this CSV path.

    Returns:
        Shuffled DataFrame with n_positive + n_negative rows.

    Raises:
        ValueError: If the ground truth does not contain enough positive or
                    negative pairs to satisfy the request.
    """
    positives = gt[gt["Match"] == True].copy()
    negatives = gt[gt["Match"] == False].copy()

    if len(positives) < n_positive:
        raise ValueError(
            f"Not enough positive pairs: requested {n_positive}, "
            f"but ground truth only has {len(positives)}."
        )
    if len(negatives) < n_negative:
        raise ValueError(
            f"Not enough negative pairs: requested {n_negative}, "
            f"but ground truth only has {len(negatives)}."
        )

    sampled_pos = _stratified_sample(positives, n_positive, seed)
    sampled_neg = _stratified_sample(negatives, n_negative, seed)

    test_set = (
        pd.concat([sampled_pos, sampled_neg], ignore_index=True)
        .sample(frac=1, random_state=seed)   # shuffle
        .reset_index(drop=True)
    )

    # Re-sequence IDs from 1
    if "ID" in test_set.columns:
        test_set["ID"] = range(1, len(test_set) + 1)
    else:
        test_set.insert(0, "ID", range(1, len(test_set) + 1))

    print(f"🎲 Sampled test set: {len(test_set)} pairs "
          f"({n_positive} positive, {n_negative} negative) | seed={seed}")
    if "Category" in test_set.columns:
        dist = test_set["Category"].value_counts().to_dict()
        print(f"   Category distribution: {dist}")

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        test_set.to_csv(out, index=False)
        print(f"💾 Saved to: {out}")

    return test_set
