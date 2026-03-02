# Semantic Checker for Open Data

A benchmarking tool that evaluates six methods for detecting semantic
equivalence between attribute names from Saxon Open Data portals.

**Methods compared:**
- Levenshtein Distance
- Jaro-Winkler Similarity
- Sentence-BERT (multilingual)
- Ollama `llama3.1:8b` — Zero-Shot, Few-Shot, Chain-of-Thought

All LLM inference runs fully on-premise via Ollama (no external API calls).

---

## Workflow

```
1. User creates / maintains ground-truth CSV
         ↓
2. Sample a balanced test set from ground truth
         ↓
3. Run all 6 methods on the test set
         ↓
4. Compare results (CSV + metrics printed to terminal)
```

---

## Prerequisites

1. **Python 3.11+**
2. **Ollama** running locally with `llama3.1:8b` pulled
   ```bash
   ollama serve            # start server (keep running in background)
   ollama pull llama3.1:8b # first-time download (~4.9 GB)
   ```

---

## Installation

```bash
cd Semantic_Checker
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Project Structure

```
Semantic_Checker/
├── main.py                        # CLI entry point (all commands)
├── run_all_baselines_csv.py       # Core benchmark runner
├── prepare_test_with_gt.py        # Legacy join helper
├── requirements.txt
│
├── baselines/
│   ├── levenshtein.py
│   ├── jaro_winkler.py
│   ├── sbert.py
│   └── ollama_prompting.py
│
├── utils/
│   ├── ground_truth_loader.py     # Load & validate ground-truth CSV
│   ├── stratified_sampler.py      # Sample balanced test sets
│   ├── data_loader.py             # Generic CSV loader for baselines
│   ├── evaluation.py              # Precision / Recall / F1
│   └── string_matching.py        # Levenshtein & Jaro-Winkler helpers
│
├── datasets/
│   ├── ground_truth/              # Manually annotated ground-truth corpora
│   │   ├── de_de_ground_truth.csv
│   │   ├── en_en_ground_truth.csv
│   │   └── mixed_ground_truth.csv
│   └── samples/test/              # Sampled test sets (output of Step 2)
│       ├── test_de_de.csv
│       ├── test_en_en.csv
│       └── test_mixed.csv
│
└── results/
    └── experiments/               # Output of Step 3
```

---

## Ground-Truth CSV Format

The ground-truth file is the **only required input** — no raw Open Data files
are needed. Create or maintain it manually.

| Column      | Required | Type    | Description                                      |
|-------------|----------|---------|--------------------------------------------------|
| Attribute1  | ✅       | string  | First attribute name                             |
| Attribute2  | ✅       | string  | Second attribute name                            |
| Match       | ✅       | boolean | `True` = semantically equivalent, `False` = not |
| Category    | optional | string  | Difficulty stratum (see below)                   |
| Confidence  | optional | string  | Annotation confidence: high / medium / low       |
| Reasoning   | optional | string  | Human annotation note                            |

**Recommended Category values:**

| Category           | Example                                         |
|--------------------|-------------------------------------------------|
| `easy_positive`    | `Straße` / `Strasse`                            |
| `conceptual_paraphrase` | `Sterbefälle` / `Todesfälle`              |
| `abbreviation`     | `PLZ` / `PostalCode`                            |
| `hard_negative`    | `Grundsteuer` / `Gewerbesteuer`                 |
| `toughest_negative`| `Bevölkerung` / `Bevölkerungsdichte`            |

---

## Step-by-Step Usage

### Step 1 — (One-off) Prepare ground truth

Edit or create a ground-truth CSV in `datasets/ground_truth/`.  
No code changes required — just a CSV file.

### Step 2 — Sample a test set

```bash
python main.py sample \
    datasets/ground_truth/de_de_ground_truth.csv \
    datasets/samples/test/test_de_de.csv \
    --n-positive 10 --n-negative 10 --seed 42
```

Options:
- `--n-positive` / `-p`  — number of positive pairs (default: 10)
- `--n-negative` / `-n`  — number of negative pairs (default: 10)
- `--seed` / `-s`        — random seed for reproducibility (default: 42)

### Step 3 — Run all baselines (string methods only)

```bash
python main.py run-all \
    datasets/samples/test/test_de_de.csv \
    --output results/experiments/test_run_de_de
```

### Step 3b — Run all baselines including Ollama

```bash
python main.py run-all \
    datasets/samples/test/test_de_de.csv \
    --output results/experiments/test_run_de_de \
    --ollama
```

### Run a single baseline

```bash
# Sentence-BERT only
python main.py baseline sbert \
    datasets/samples/test/test_de_de.csv \
    results/experiments/sbert/de_de.csv

# Ollama few-shot only
python main.py baseline ollama-fewshot \
    datasets/samples/test/test_de_de.csv \
    results/experiments/ollama/de_de_fewshot.csv
```

Available methods: `levenshtein`, `jaro-winkler`, `sbert`,
`ollama`, `ollama-zeroshot`, `ollama-fewshot`, `ollama-cot`

---

## Reproducing the Paper Results

```bash
# DE-DE
python main.py sample datasets/ground_truth/de_de_ground_truth.csv \
    datasets/samples/test/test_de_de.csv --seed 42
python main.py run-all datasets/samples/test/test_de_de.csv \
    --output results/experiments/test_run_de_de --ollama

# EN-EN
python main.py sample datasets/ground_truth/en_en_ground_truth.csv \
    datasets/samples/test/test_en_en.csv --seed 42
python main.py run-all datasets/samples/test/test_en_en.csv \
    --output results/experiments/test_run_en_en --ollama

# Mixed
python main.py sample datasets/ground_truth/mixed_ground_truth.csv \
    datasets/samples/test/test_mixed.csv --seed 42
python main.py run-all datasets/samples/test/test_mixed.csv \
    --output results/experiments/test_run_mixed --ollama
```

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| `Ollama is not running` | Run `ollama serve` in a separate terminal |
| `Model not found` | Run `ollama pull llama3.1:8b` |
| `Missing column 'Match'` | Add a `Match` column (True/False) to your ground-truth CSV |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` inside the `.venv` |

---

## Environment Variables

| Variable       | Default           | Description                   |
|----------------|-------------------|-------------------------------|
| `OLLAMA_MODEL` | `llama3.1:8b`     | Override the Ollama model name |
