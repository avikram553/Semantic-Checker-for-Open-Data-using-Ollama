# BERT and SBERT Evaluation Guide

## 🎯 Understanding Your Setup

### What You Have
- ✅ **Pre-trained BERT** (`baselines/bert_base.py`) - English-only semantic similarity
- ✅ **Pre-trained SBERT** (`baselines/sbert.py`) - Multilingual semantic similarity
- ✅ **Ground truth dataset** - Labeled attribute pairs for evaluation

### What You DON'T Need
- ❌ **Training from scratch** - Models are already pre-trained
- ❌ **Fine-tuning** - Not required for zero-shot evaluation (standard approach)
- ❌ **GPU** - Can run on CPU (slower but works)

---

## 📊 Running Evaluations (Zero-Shot)

### Option 1: Run Individual Baselines

#### 1. BERT Base Evaluation
```bash
cd /Users/adityavikram/Desktop/Research_Internship_2025/Semantic_Checker
source .venv/bin/activate

# Run BERT evaluation
python -m baselines.bert_base
```

**What it does:**
- Loads pre-trained `bert-base-uncased` model
- Computes [CLS] token embeddings for each attribute
- Calculates cosine similarity between embeddings
- Outputs: `data/bert_base_results.csv`

#### 2. SBERT Evaluation
```bash
# Run SBERT evaluation
python -m baselines.sbert
```

**What it does:**
- Loads pre-trained `paraphrase-multilingual-MiniLM-L12-v2`
- Computes sentence embeddings optimized for similarity
- Better for cross-lingual and semantic matching
- Outputs: `data/sbert_results.csv`

#### 3. Levenshtein & Jaro-Winkler (String-based baselines)
```bash
# Run string similarity baselines
python -m baselines.levenshtein
python -m baselines.jaro_winkler
```

---

### Option 2: Run All Baselines at Once

```bash
cd /Users/adityavikram/Desktop/Research_Internship_2025/Semantic_Checker
source .venv/bin/activate

# Run all baseline comparisons
python run_all_baselines_csv.py
```

**This will:**
- Run Levenshtein, Jaro-Winkler, BERT, and SBERT
- Save individual results to CSV files
- Generate comparison metrics
- Create summary report

---

## 📈 Evaluation Metrics Computed

For each method, you'll get:
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: Harmonic mean of precision and recall
- **Accuracy**: (TP + TN) / Total
- **Confusion Matrix**: TP, FP, TN, FN counts

---

## 🔬 Advanced: Fine-Tuning (Optional - For Better Performance)

### When to Fine-Tune?
Fine-tune if you want to:
- ✅ Improve performance on your specific domain (open data attributes)
- ✅ Adapt to your naming conventions
- ✅ Publish novel results in your research

### When NOT to Fine-Tune?
- ❌ Just benchmarking existing methods (your current goal)
- ❌ Limited training data (< 1000 pairs)
- ❌ Limited computational resources

### Fine-Tuning Steps (If Needed)

#### Step 1: Prepare Training Data
Your ground truth pairs need to be split:
- **Training set**: 70-80% of pairs
- **Validation set**: 10-15% of pairs
- **Test set**: 10-15% of pairs (never seen during training)

#### Step 2: Fine-Tune BERT/SBERT

I can create a fine-tuning script if needed. Here's the general approach:

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Load pre-trained model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Prepare training data
train_examples = []
for _, row in train_df.iterrows():
    score = 1.0 if row['match'] else 0.0
    train_examples.append(InputExample(texts=[row['attribute1'], row['attribute2']], label=score))

# Create DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Define loss function
train_loss = losses.CosineSimilarityLoss(model)

# Fine-tune
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=4,
    warmup_steps=100
)

# Save fine-tuned model
model.save('models/sbert-finetuned-attributes')
```

---

## 🎓 Recommended Approach for Your Research

### Phase 1: Zero-Shot Evaluation (Current - RECOMMENDED)
1. ✅ Run all baselines with pre-trained models
2. ✅ Compare performance metrics
3. ✅ Analyze results (which methods work best?)
4. ✅ Write research paper comparing approaches

**Advantages:**
- Standard approach in academic research
- Fair comparison (all models evaluated equally)
- Reproducible results
- No risk of overfitting

### Phase 2: Fine-Tuning (Optional - Advanced)
Only if you want to:
- Show improvement over baselines
- Propose a domain-specific model
- Publish method contribution paper

---

## 🚀 Quick Start - Run Everything Now

```bash
# 1. Activate virtual environment
cd /Users/adityavikram/Desktop/Research_Internship_2025/Semantic_Checker
source .venv/bin/activate

# 2. Check if required packages are installed
pip install transformers torch sentence-transformers scikit-learn

# 3. Run all baselines
python run_all_baselines_csv.py

# 4. Generate visualizations (simple tables)
python visualize_results.py --tables

# 5. View results
ls -lh data/*_results.csv
ls -lh visualizations/*.csv
```

---

## 📊 Expected Output Files

After running evaluations:

```
data/
├── levenshtein_results.csv      # String distance baseline
├── jaro_winkler_results.csv     # String similarity baseline
├── bert_base_results.csv        # BERT embeddings (English)
└── sbert_results.csv            # SBERT embeddings (Multilingual)

visualizations/
├── levenshtein_true_positives.csv
├── levenshtein_false_negatives.csv
├── bert_base_true_positives.csv
├── sbert_true_positives.csv
└── ... (similar for all methods)
```

---

## 💡 Key Insights

### BERT vs SBERT Comparison

| Aspect | BERT Base | SBERT |
|--------|-----------|-------|
| **Training** | General language modeling | Fine-tuned for similarity |
| **Speed** | Slower | Faster |
| **Multilingual** | No (English only) | Yes |
| **Semantic Quality** | Good | Better for similarity tasks |
| **Your Use Case** | Baseline comparison | Recommended method |

### Expected Performance (Based on Literature)

For attribute name matching:
- **Levenshtein**: F1 ~0.40-0.50 (character-level)
- **Jaro-Winkler**: F1 ~0.70-0.80 (better for prefixes)
- **BERT Base**: F1 ~0.75-0.85 (semantic, English only)
- **SBERT**: F1 ~0.80-0.90 (best semantic, multilingual)

---

## ❓ FAQ

### Q: Do I need a GPU?
**A:** No, but it's slower on CPU. For your dataset size (~300-1200 pairs), CPU is fine (10-30 minutes).

### Q: Should I fine-tune?
**A:** Not required for research comparison. Fine-tune only if you want to publish a novel method.

### Q: Which model should I use?
**A:** For your multilingual open data scenario, **SBERT** is the best choice. BERT is good as a baseline comparison.

### Q: How do I cite these models in my paper?
**A:**
- BERT: Devlin et al., 2019, "BERT: Pre-training of Deep Bidirectional Transformers"
- SBERT: Reimers & Gurevych, 2019, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"

---

## 📚 Next Steps

1. **Run evaluations** using the commands above
2. **Analyze results** - which method performs best?
3. **Visualize** - use the simple tables you already created
4. **Write paper** - compare methods, discuss results
5. **(Optional)** Fine-tune if you need better performance

Would you like me to:
1. Create a fine-tuning script?
2. Run the evaluations now?
3. Create a results comparison script?
