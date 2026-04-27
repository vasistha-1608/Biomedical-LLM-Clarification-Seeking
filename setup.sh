#!/bin/bash
# setup.sh — Environment setup for Biomedical LLM Clarification-Seeking Pipeline
# Usage: bash setup.sh

set -e

echo "============================================"
echo "Setting up Biomedical LLM Clarification-Seeking"
echo "============================================"


curl -sS https://bootstrap.pypa.io/get-pip.py | /usr/bin/python3.10
/usr/bin/python3.10 -m pip install ipykernel -U
# Core ML frameworks
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers

# Data processing & analysis
pip install numpy pandas scikit-learn

# NLP
pip install spacy
python -m spacy download en_core_web_sm

# Visualization
pip install matplotlib seaborn

# PubMed API access (for text_generation.ipynb)
pip install biopython

# Jupyter support
pip install jupyter ipywidgets

pip install bert-score sacrebleu rouge-score

# Download model weights (cached for offline use)
echo ""
echo "============================================"
echo "Pre-downloading model weights..."
echo "============================================"
python -c "
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch

print('Downloading BioBERT...')
AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.2')
AutoModelForSequenceClassification.from_pretrained('dmis-lab/biobert-base-cased-v1.2', num_labels=4)

print('Downloading PubMedBERT...')
AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
AutoModelForSequenceClassification.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext', num_labels=4)

print('Downloading BioMistral-7B...')
AutoTokenizer.from_pretrained('BioMistral/BioMistral-7B')
AutoModelForCausalLM.from_pretrained('BioMistral/BioMistral-7B', torch_dtype=torch.float16, use_safetensors=False)

print('All models cached successfully!')
"

echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
echo "Run notebooks in order:"
echo "  1. parse_emb.ipynb"
echo "  2. masking_data.ipynb"
echo "  3. data_tokenization.ipynb"
echo "  4. biobert.ipynb / pubmedbert.ipynb"
echo "  5. untrained_baseline.ipynb"
echo "  6. text_generation.ipynb"
echo "============================================"
