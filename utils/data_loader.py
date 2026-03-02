
import pandas as pd
from pathlib import Path


def load_pair_data(filepath: str) -> pd.DataFrame:
    """
    Load attribute pairs from a CSV file into a unified DataFrame.

    The file must be a CSV with at least the columns:
        Attribute1, Attribute2
    and optionally:
        Match, ID, Category, Confidence, Reasoning

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame with normalised columns. Match column is boolean when present.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a CSV or is missing required columns.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if path.suffix.lower() != ".csv":
        raise ValueError(
            f"Only CSV files are supported, got: {path.suffix}"
        )

    # Try UTF-8 first, fall back to latin-1 for files with umlauts
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Could not decode file with any supported encoding: {filepath}")

    # Validate required columns
    for col in ("Attribute1", "Attribute2"):
        if col not in df.columns:
            raise ValueError(
                f"CSV is missing required column '{col}'. "
                f"Found columns: {list(df.columns)}"
            )

    # Normalise Match to boolean when present
    if "Match" in df.columns:
        if df["Match"].dtype != bool:
            df["Match"] = (
                df["Match"]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"true": True, "1": True, "yes": True,
                      "false": False, "0": False, "no": False})
            )

    return df
