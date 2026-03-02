"""
utils/ground_truth_loader.py
-----------------------------
Loads and validates a manually curated ground-truth CSV file.

Expected schema (required columns):
    Attribute1  – first attribute name of the pair
    Attribute2  – second attribute name of the pair
    Match       – boolean equivalence label (True / False)

Optional columns (used for stratified sampling):
    Category    – difficulty stratum (e.g. easy_positive, abbreviation, hard_negative)
    Confidence  – annotation confidence (high / medium / low)
    Reasoning   – human annotation note

Usage:
    from utils.ground_truth_loader import load_ground_truth
    df = load_ground_truth("datasets/ground_truth/de_de_ground_truth.csv")
"""

import pandas as pd
from pathlib import Path


REQUIRED_COLUMNS = ["Attribute1", "Attribute2", "Match"]
OPTIONAL_COLUMNS = ["Category", "Confidence", "Reasoning"]


class GroundTruthSchemaError(ValueError):
    """Raised when the ground-truth file is missing required columns."""
    pass


def load_ground_truth(filepath: str) -> pd.DataFrame:
    """
    Load and validate a ground-truth CSV file.

    Args:
        filepath: Path to the ground-truth CSV file.

    Returns:
        DataFrame with at least columns [Attribute1, Attribute2, Match].
        Match column is guaranteed to be boolean dtype.

    Raises:
        FileNotFoundError: If the file does not exist.
        GroundTruthSchemaError: If any required column is missing.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {filepath}")

    if path.suffix.lower() != ".csv":
        raise ValueError(
            f"Expected a CSV file, got: {path.suffix}. "
            "Only CSV ground-truth files are supported."
        )

    # Try UTF-8 first, fall back to latin-1 for umlauts
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Could not decode file with any supported encoding: {filepath}")

    # ── Schema validation ────────────────────────────────────
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise GroundTruthSchemaError(
            f"Ground-truth file is missing required column(s): {missing}\n"
            f"Found columns: {list(df.columns)}\n"
            f"Required columns: {REQUIRED_COLUMNS}"
        )

    # ── Normalise Match to boolean ───────────────────────────
    if df["Match"].dtype != bool:
        df["Match"] = (
            df["Match"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "1": True, "yes": True,
                  "false": False, "0": False, "no": False})
        )
        if df["Match"].isna().any():
            bad = df[df["Match"].isna()][["Attribute1", "Attribute2"]].head(5)
            raise GroundTruthSchemaError(
                f"Match column contains unrecognised values. "
                f"Use True/False. Problem rows:\n{bad}"
            )

    # ── Normalise attribute name whitespace ──────────────────
    df["Attribute1"] = df["Attribute1"].astype(str).str.strip()
    df["Attribute2"] = df["Attribute2"].astype(str).str.strip()

    # ── Add Category if missing (needed for stratified sampling) ─
    if "Category" not in df.columns:
        df["Category"] = "unspecified"

    # ── Summary ─────────────────────────────────────────────
    n_pos = df["Match"].sum()
    n_neg = (~df["Match"]).sum()
    print(f"✅ Loaded ground truth: {len(df)} pairs "
          f"({n_pos} positive, {n_neg} negative) from {path.name}")
    if "Category" in df.columns and df["Category"].nunique() > 1:
        print(f"   Categories: {sorted(df['Category'].unique())}")

    return df
