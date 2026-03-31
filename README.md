# Biomedical LLM Clarification-Seeking Pipeline

> **A decoupled 5-stage detect-and-clarify system that teaches LLMs to ask targeted clarification questions instead of hallucinating answers when clinical context is incomplete.**


## Problem

LLMs used in medical question answering will confidently generate answers even when critical clinical information is missing — leading to potentially harmful clinical decisions. Standard approaches either guess (risking hallucination) or refuse entirely (providing no value). We build a system that does neither: it **detects what's missing and asks the right question**.

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌──────────────┐    ┌──────────────┐
│  Stage 1:   │    │  Stage 2:    │    │    Stage 3:     │    │  Stage 4:    │    │  Stage 5:    │
│  Controlled │───▶│  Fine-tune   │───▶│  Classification │───▶│  Dynamic     │───▶│  Decoder     │
│  PICO       │    │  BioBERT     │    │  (Gatekeeper)   │    │  Prompt      │    │  Generation  │
│  Masking    │    │  Encoder     │    │                 │    │  Handshake   │    │              │
└─────────────┘    └──────────────┘    └────────┬────────┘    └──────────────┘    └──────┬───────┘
                                                │                                        │
                                         Complete? ──Yes──▶ Standard EBM Extraction      │
                                                │                                        │
                                          No (Missing P/I/O)                             │
                                                └──────────────▶ Targeted Clarification ◀┘
                                                                  Question
```

The key insight is **decoupling**: a small, supervised encoder (BioBERT, 110M params) handles ambiguity detection deterministically, while a larger decoder (BioMistral-7B) only generates text under strict prompt constraints. This prevents the decoder from masking uncertainty with hallucinated clinical details.

## Key Results

### Task 1: Missing-Slot Detection

| Model | Parameters | Setting | Accuracy |
|-------|-----------|---------|----------|
| GPT-4o | ~1.8T (est.) | Few-shot | 83.1% |
| **BioBERT (ours)** | **110M** | **Supervised** | **83.0%** |
| BioMistral-7B | 7B | Few-shot | 66.7% |
| Phi-3-mini | 3.8B | Few-shot | 51.9% |
| SmolLM2 | 1.7B | Few-shot | 29.6% |

**Our fine-tuned BioBERT encoder matches GPT-4o's few-shot accuracy at 1/1000th the model size.**

Per-class F1: Complete (0.93) | Missing P (0.72) | Missing I (0.70) | Missing O (0.75) — Macro-F1: **0.78**

### Task 2: Clarification Question Quality (n=141)

| Criterion | Mean Score |
|-----------|-----------|
| Slot Targeting | 4.75 / 5 |
| Specificity | 4.36 / 5 |
| No Assumptions (Hallucination Avoidance) | 4.87 / 5 |

## Dataset

Built from the **EBM-NLP corpus** (Nye et al., 2018):
- **4,981 clinical abstracts** with token-level P/I/O annotations
- Transformed into a sequence classification dataset via controlled masking
- Balanced split: 50.01% complete / 49.99% masked
- Uniform missing-slot distribution: Outcomes (35.26%), Populations (32.73%), Interventions (32.01%)
- Train: 4,792 | Test: 189

## Project Structure

```
├── biobert.ipynb                # BioBERT fine-tuning for 4-class PICO classification
├── data_tokenization.ipynb      # EBM-NLP preprocessing and tokenization
├── masking_data.ipynb           # Controlled PICO slot masking pipeline
├── masked_data_eda.ipynb        # Exploratory data analysis on masked dataset
├── parse_emb.ipynb              # Embedding extraction and analysis
├── text_generation.ipynb        # Decoder generation experiments
├── inference_wrapper.ipynb      # End-to-end pipeline inference
├── inference_wrapper.py         # Production inference script
├── prompt_generator.py          # Dynamic prompt construction (Stage 4)
└── saved_biobert_pico_model/    # Fine-tuned BioBERT checkpoint
```

## Setup

```bash
git clone https://github.com/vasistha-1608/Biomedical-LLM-Clarification-Seeking.git
cd Biomedical-LLM-Clarification-Seeking
pip install torch transformers datasets scikit-learn spacy
python -m spacy download en_core_web_sm
```

## Usage

```python
from inference_wrapper import run_pipeline

# Provide a clinical abstract (potentially incomplete)
result = run_pipeline("A randomized controlled trial evaluated the efficacy of ciprofloxacin suspension...")

# Returns:
# - classification: "Missing O" (or "Complete", "Missing P", "Missing I")
# - response: Targeted clarification question OR structured EBM extraction
```

## Research Questions

- **RQ1:** Can an LLM detect which PICO slot is missing from a masked abstract? → **Yes, 83% accuracy with supervised BioBERT**
- **RQ2:** Can it ask a targeted clarifying question rather than guessing? → **Yes, 4.75/5 slot-targeting score**
- **RQ3:** Does the pipeline give better answers with complete information than few-shot prompting? → **In progress**

## Known Limitations & Ongoing Work

- **OOD generalization gap:** The encoder shows a systematic "Missing P" bias on real unmasked PubMed abstracts (7/10 misclassified). We're experimenting with span-level masking to address this.
- **Self-evaluation bias:** Task 2 uses GPT-4o as both generator and judge. Human evaluation is planned.
- **Decoder comparison:** Currently evaluating BioMistral-7B and Llama-3.1-8B-Instruct as open-source decoder alternatives.

## Tech Stack

- **Encoder:** BioBERT (fine-tuned, 110M params)
- **Decoder:** BioMistral-7B / GPT-4o
- **Framework:** PyTorch, HuggingFace Transformers
- **Dataset:** EBM-NLP (Nye et al., 2018)
- **NLP:** spaCy (sentence boundary detection)

## Contributors

- [Vasistha Eranki](https://github.com/vasistha-1608) — University of Illinois Urbana-Champaign
- Mihir Sahasrabudhe — University of Illinois Urbana-Champaign

## Citation

If you find this work useful, please cite:

```bibtex
@misc{eranki2026biomedical,
  title={LLMs Seeking Clarification in Medical Question Answering},
  author={Eranki, Vasistha and Sahasrabudhe, Mihir},
  year={2026},
  institution={University of Illinois Urbana-Champaign}
}
```
