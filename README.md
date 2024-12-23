# Consensus Technical Assessment: Language Model Fine-Tuning, Evaluation, and Optimization

## Overview

This repository implements the Consensus Technical Assessment to fine-tune and optimize a language model for scientific question answering. The task involves:

1. Fine-tuning a T5-based language model using the training dataset (`data/input_data/training_data.csv`).
2. Running inference on an evaluation dataset (`data/input_data/inference_data.csv`) to generate answers.
3. Ensuring inference meets the speed requirement: **20 text inputs in under 3 seconds**.
4. Evaluating model performance using robust metrics.

---

## Deliverables

### Training Code
- **Script**: `models/train.py`
- Preprocesses input data by cleaning and formatting.
- Fine-tunes a pre-trained `T5-small` model.
- Saves the best-performing model and checkpoints.

### Inference Code
- **Script**: `inference.py`
- Preprocesses and formats questions and contexts.
- Runs batch inference on 20 inputs.
- Measures inference latency and ensures it meets the < 3-second requirement.

### Evaluation Metrics
- **Script**: `models/evaluation.py`
- Metrics Used:
  1. **BLEU**: Measures n-gram overlap for lexical similarity.
  2. **ROUGE**: Evaluates unigram, bigram, and longest sequence overlap.
  3. **Exact Match**: Checks strict word-for-word correctness.
  4. **BERTScore**: Assesses semantic similarity using contextual embeddings.

---

## Why These Metrics?

| Metric         | Purpose                                   | Justification                                      |
|----------------|-------------------------------------------|--------------------------------------------------|
| **BLEU**       | Measures n-gram precision.               | Ensures answers contain key terms from references. |
| **ROUGE**      | Measures word/phrase overlap.            | Evaluates content coverage for longer answers.    |
| **Exact Match**| Checks strict correctness.               | Useful for tasks requiring exact technical phrasing. |
| **BERTScore**  | Measures semantic similarity.            | Captures paraphrases and high-level meaning.       |

---

## Evaluation Results

### Speed Test
- **Batch Size**: 20 inputs.
- **Elapsed Time**: [To be filled once the script completes].

### Evaluation Metrics
| Metric           | Value   |
|-------------------|---------|
| **Average BLEU**  | [TBD]   |
| **Average ROUGE** | [TBD]   |
| **Exact Match**   | [TBD]   |
| **BERTScore**     | [TBD]   |

---

## How to Run

```bash
python inference.py
```

```bash
python models/evaluation.py
```
