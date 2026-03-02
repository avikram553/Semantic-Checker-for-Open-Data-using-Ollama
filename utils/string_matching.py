"""
utils/string_matching.py
-------------------------
Pure-Python string similarity functions used by the Levenshtein and
Jaro-Winkler baselines.

All functions are case-insensitive by convention: callers should pass
lower-cased strings, or use the normalised wrappers (levenshtein_similarity,
jaro_winkler_similarity) which lower-case internally.

Public API:
    levenshtein_distance(s1, s2)       -> int
    levenshtein_similarity(s1, s2)     -> float  [0.0, 1.0]
    jaro_similarity(s1, s2)            -> float  [0.0, 1.0]
    jaro_winkler_similarity(s1, s2)    -> float  [0.0, 1.0]
"""


def levenshtein_distance(s1: str, s2: str) -> int:
    """Return the edit distance between *s1* and *s2*.

    Uses the standard dynamic-programming algorithm (Wagner–Fischer).
    Complexity: O(|s1| × |s2|) time, O(|s2|) space.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def levenshtein_similarity(s1: str, s2: str) -> float:
    """Return a normalised similarity score in [0.0, 1.0].

    Defined as ``1 - (edit_distance / max_length)``.  Both strings are
    lower-cased before comparison.  Returns 1.0 for two empty strings.
    """
    distance = levenshtein_distance(s1.lower(), s2.lower())
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return 1.0 - (distance / max_len)


def jaro_similarity(s1: str, s2: str) -> float:
    """Return the Jaro similarity between *s1* and *s2* in [0.0, 1.0].

    Two characters are considered matching if they are within
    ``floor(max(|s1|, |s2|) / 2) - 1`` positions of each other.
    """
    if len(s1) == 0 and len(s2) == 0:
        return 1.0
    if len(s1) == 0 or len(s2) == 0:
        return 0.0

    match_distance = max(len(s1), len(s2)) // 2 - 1
    if match_distance < 1:
        match_distance = 1

    s1_matches = [False] * len(s1)
    s2_matches = [False] * len(s2)
    matches = 0
    transpositions = 0

    for i in range(len(s1)):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len(s2))

        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len(s1)):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    jaro = (
        matches / len(s1)
        + matches / len(s2)
        + (matches - transpositions / 2) / matches
    ) / 3.0
    return jaro


def jaro_winkler_similarity(s1: str, s2: str, prefix_weight: float = 0.1) -> float:
    """Return the Jaro-Winkler similarity between *s1* and *s2* in [0.0, 1.0].

    Jaro-Winkler boosts the Jaro score for strings that share a common
    prefix (up to 4 characters).  *prefix_weight* controls the boost
    magnitude (standard value: 0.1).
    """
    jaro = jaro_similarity(s1, s2)

    prefix = 0
    for i in range(min(len(s1), len(s2), 4)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    return jaro + (prefix * prefix_weight * (1.0 - jaro))
