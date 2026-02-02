#!/usr/bin/env python3
"""
Generate a clean, unique 1000-pair ground truth dataset for semantic similarity analysis.
"""

import os
import glob
import pandas as pd
import random
from typing import List, Set, Tuple, Dict

# Configuration
DATA_DIR = "Semantic_Checker/data"
OUTPUT_FILE = os.path.join(DATA_DIR, "clean_ground_truth_1000.csv")
TARGET_PAIRS = 1000

# Dictionary for generating Positive pairs (German <-> English, Synonyms)
# Base pairs that we know are semantically equivalent
POSITIVE_MAPPINGS = {
    "lat": ["latitude", "breitengrad", "y_koordinate", "north"],
    "lon": ["longitude", "laengengrad", "längengrad", "x_koordinate", "east"],
    "id": ["identifier", "kennung", "schluessel", "nummer", "objectid", "fid", "gid"],
    "name": ["bezeichnung", "titel", "straßenname", "name_amtl"],
    "date": ["datum", "zeitpunkt", "erstellungsdatum", "vondatum", "bisdatum"],
    "address": ["adresse", "anschrift", "standort"],
    "street": ["straße", "strasse", "str", "streetname"],
    "housenumber": ["hausnummer", "hnr", "haus_nr"],
    "city": ["stadt", "ort", "gemeinde"],
    "zip": ["plz", "postleitzahl", "postal_code"],
    "district": ["stadtteil", "ortsteil", "bezirk", "wahlbezirk"],
    "area": ["flaeche", "fläche", "qm", "size"],
    "count": ["anzahl", "menge", "total"],
    "type": ["typ", "art", "kategorie"],
    "status": ["zustand", "phase"],
    "geometry": ["the_geom", "wkt", "shape", "geometrie"],
    "description": ["beschreibung", "bemerkung", "info"]
}

def get_all_csv_files(root_dir: str) -> List[str]:
    """recursively find all csv files"""
    return glob.glob(os.path.join(root_dir, "**/*.csv"), recursive=True)

def extract_attributes(csv_files: List[str]) -> Set[str]:
    """Extract all unique column headers from csv files"""
    attributes = set()
    print(f"Scanning {len(csv_files)} files for attributes...")
    
    for f in csv_files:
        if "ground_truth" in f or "results" in f:
            continue
            
        try:
            # Read only header, try multiple encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            for encoding in encodings:
                try:
                    df = pd.read_csv(f, nrows=0, encoding=encoding, sep=None, engine='python')
                    clean_cols = []
                    for c in df.columns:
                        val = str(c).strip()
                        # Filter out garbage headers
                        if not val: continue 
                        if "Unnamed" in val: continue
                        if val.replace('.', '').isdigit(): continue # Pure numbers
                        if "Qu." in val: continue # Date values misread as headers
                        clean_cols.append(val)
                        
                    attributes.update(clean_cols)
                    break 
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    # If it's not an encoding error, standard pandas error might happen
                    continue
        except Exception as e:
            print(f"Skipping {f}: {e}")
            
    print(f"Found {len(attributes)} unique attributes.")
    return attributes

def generate_positives(attributes: List[str], count: int) -> Set[Tuple[str, str]]:
    """Generate positive pairs using exact matches and dictionary mappings"""
    positives = set()
    
    # 1. Dictionary-based generation
    # Create reverse map for O(1) lookups: specific_term -> concept_key
    term_to_key = {}
    for key, synonyms in POSITIVE_MAPPINGS.items():
        term_to_key[key] = key
        for syn in synonyms:
            term_to_key[syn] = key
            
    # Find attributes that match our keys
    matched_attrs = {key: [] for key in POSITIVE_MAPPINGS}
    
    for attr in attributes:
        attr_lower = attr.lower()
        # Direct match
        if attr_lower in term_to_key:
            matched_attrs[term_to_key[attr_lower]].append(attr)
            continue
            
        # Substring match (e.g. "strasse_name" matches "strasse")
        for key, synonyms in POSITIVE_MAPPINGS.items():
            if key in attr_lower or any(s in attr_lower for s in synonyms):
                matched_attrs[key].append(attr)
                break
    
    # Generate pairs from within the same concept bucket
    for key, attrs in matched_attrs.items():
        if len(attrs) < 2:
            continue
        # Create all combinations
        for i in range(len(attrs)):
            for j in range(i + 1, len(attrs)):
                positives.add((attrs[i], attrs[j]))
                
    # 2. Exact matches (simulated)
    # Since we have a list of unique attributes, we can't do exact matches of the same string
    # unless we treat them as "same concept". 
    # But strictly speaking, (A, A) is trivial. We want (A_file1, A_file2).
    # Since we flattened matches to a set, we don't have source info.
    # Instead, let's look for slight case variations if any exist in the raw set?
    
    return positives

def generate_negatives(attributes: List[str], count: int) -> Set[Tuple[str, str]]:
    """Generate negative pairs: hard negatives and random negatives"""
    negatives = set()
    attrs_list = list(attributes)
    
    attempts = 0
    while len(negatives) < count and attempts < count * 10:
        attempts += 1
        a1 = random.choice(attrs_list)
        a2 = random.choice(attrs_list)
        
        if a1 == a2:
            continue
            
        # Check if they look like a positive pair (heuristically)
        # If they share significant substrings, they might be positive or hard negative
        # For now, we assume if they are NOT in our positive logic, they are negative.
        # But we want HARD negatives too.
        
        # Hard Negative Heuristic: Share a substring but are different
        # e.g. "Start_Date" vs "End_Date"
        
        pair = tuple(sorted((a1, a2)))
        negatives.add(pair)
        
    return negatives

def main():
    csv_files = get_all_csv_files(DATA_DIR)
    attributes = extract_attributes(csv_files)
    
    if len(attributes) < 10:
        print("Not enough attributes found!")
        return

    # Generate candidates
    print("Generating pairs...")
    potential_positives = generate_positives(list(attributes), TARGET_PAIRS)
    print(f"Generated {len(potential_positives)} potential positive pairs.")
    
    # We want roughly 50/50 split
    target_pos = TARGET_PAIRS // 2
    target_neg = TARGET_PAIRS - target_pos
    
    final_pairs = []
    
    # Select Positives
    pos_list = list(potential_positives)
    if len(pos_list) > target_pos:
        selected_pos = random.sample(pos_list, target_pos)
    else:
        selected_pos = pos_list
        # Fill semantic checking gap later if needed
        
    for p in selected_pos:
        final_pairs.append({
            "Attribute1": p[0], 
            "Attribute2": p[1], 
            "Match": True, 
            "Type": "positive"
        })
        
    # Select Negatives
    # We need to ensure we don't accidentally pick a positive as a negative
    # Simpler approach: Randomly pick, and if it's not in potential_positives, it's a negative.
    
    neg_needed = TARGET_PAIRS - len(final_pairs)
    generated_negs = set()
    
    while len(generated_negs) < neg_needed:
        a1 = random.choice(list(attributes))
        a2 = random.choice(list(attributes))
        
        if a1 == a2: continue
        
        pair_key = tuple(sorted((a1, a2)))
        if pair_key in potential_positives: continue
        if pair_key in generated_negs: continue
        
        generated_negs.add(pair_key)
        
        # Determine strictness (simple random for now)
        final_pairs.append({
            "Attribute1": a1,
            "Attribute2": a2,
            "Match": False,
            "Type": "negative"
        })
        
    # Shuffle
    random.shuffle(final_pairs)
    
    # Add IDs
    for i, row in enumerate(final_pairs, 1):
        row['ID'] = i
        
    # Save
    df = pd.DataFrame(final_pairs)
    # Ensure column order
    cols = ['ID', 'Attribute1', 'Attribute2', 'Match', 'Type']
    df = df[cols]
    
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Successfully saved {len(df)} pairs to {OUTPUT_FILE}")
    print(f"Breakdown: {len(df[df['Match']==True])} Positives, {len(df[df['Match']==False])} Negatives")

if __name__ == "__main__":
    main()
