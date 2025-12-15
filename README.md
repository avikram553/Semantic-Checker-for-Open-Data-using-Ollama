# Semantic Checker

A Python-based semantic checker using local LLM (Ollama with Llama 3.1 8B) for semantic similarity analysis.

## Features

- **Local LLM Integration**: Uses Ollama with Llama 3.1 8B for semantic analysis
- **CSV Data Processing**: Load and process text pairs from CSV files
- **Batch Analysis**: Analyze multiple text pairs efficiently
- **Evaluation Metrics**: Calculate accuracy, precision, recall, and F1-score
- **Embedding Support**: Optional sentence-transformers for baseline comparison

## Technology Stack

- **Python 3.11+**: Main programming language
- **Ollama 0.3.x**: Local LLM runtime
- **Llama 3.1 8B**: Primary language model
- **pandas 2.1+**: Data manipulation and CSV parsing
- **sentence-transformers**: Embedding-based baseline
- **scikit-learn**: Evaluation metrics
- **requests**: Ollama API communication

## Prerequisites

1. **Python 3.11 or higher** installed on your system
2. **Ollama** installed and running locally
   - Install from: https://ollama.ai
   - Pull Llama 3.1 8B model: `ollama pull llama3.1:8b`

## Installation

1. Clone or navigate to the repository:
```bash
cd /path/to/Semantic_Checker
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
Semantic_Checker/
├── src/
│   ├── __init__.py
│   ├── ollama_client.py      # Ollama API client
│   ├── semantic_analyzer.py   # Main semantic analysis logic
│   ├── data_processor.py      # CSV data handling
│   └── evaluator.py           # Performance evaluation
├── tests/                     # Unit tests
├── data/                      # Data files
├── config/                    # Configuration files
├── main.py                    # Main entry point
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Usage

### Starting Ollama

Before running the semantic checker, ensure Ollama is running:

```bash
ollama serve
```

In another terminal, verify the model is available:

```bash
ollama pull llama3.1:8b
```

### Creating Sample Data

Create a sample CSV file with text pairs:

```python
from src.data_processor import DataProcessor

processor = DataProcessor("data/sample.csv")
processor.create_sample_data("data/sample.csv", num_samples=10)
```

### Running Semantic Analysis

```bash
python main.py --input data/sample.csv --output results.csv --mode analyze
```

### Command Line Arguments

- `--input`: Path to input CSV file (required)
- `--output`: Path to output CSV file (default: results.csv)
- `--model`: Ollama model to use (default: llama3.1:8b)
- `--mode`: Operation mode - `analyze`, `evaluate`, or `both` (default: both)

### Input CSV Format

Your input CSV should have at least these columns:
- `text1`: First text in the pair
- `text2`: Second text in the pair
- `label`: (Optional) True label for evaluation (SIMILAR or DIFFERENT)

Example:
```csv
text1,text2,label
"The cat sat on the mat","A feline rested on the rug",SIMILAR
"I love programming","The weather is nice",DIFFERENT
```

## API Usage

```python
from src.ollama_client import OllamaClient
from src.semantic_analyzer import SemanticAnalyzer

# Initialize client
client = OllamaClient(model="llama3.1:8b")

# Create analyzer
analyzer = SemanticAnalyzer(client)

# Analyze a pair
result = analyzer.analyze_pair(
    "The cat sat on the mat",
    "A feline rested on the rug"
)

print(result['prediction'])  # SIMILAR or DIFFERENT
print(result['explanation'])  # LLM explanation
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ tests/
```

### Linting

```bash
flake8 src/ tests/
mypy src/
```

## Configuration

Environment variables can be set in a `.env` file:

```
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
```

## Evaluation Metrics

The evaluator calculates:
- **Accuracy**: Overall correctness
- **Precision**: Positive predictive value
- **Recall**: Sensitivity
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed prediction breakdown

## Troubleshooting

### Ollama Not Running
```
Error: Ollama is not running. Please start Ollama first.
```
**Solution**: Start Ollama with `ollama serve`

### Model Not Found
```
Error: Model not found
```
**Solution**: Pull the model with `ollama pull llama3.1:8b`

### Import Errors
```
ModuleNotFoundError: No module named 'pandas'
```
**Solution**: Install dependencies with `pip install -r requirements.txt`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

MIT License

## Contact

For questions or issues, please open an issue on the repository.

## Acknowledgments

- Ollama team for the local LLM runtime
- Meta AI for Llama 3.1
- Sentence-Transformers team for embedding models
