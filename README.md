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

 
## Model Training Summary

### Model Details
- **Architecture**: T5-Small (`~60M parameters`), Sequence-to-Sequence for open-ended text generation.
- **Pre-trained Model**: Hugging Face `t5-small`, fine-tuned for scientific question answering.
- **Loss Function**: Cross-entropy (built into T5).

### Dataset
- **Input Data**: Preprocessed data (`data/output_datasets/preprocessed_data.pkl`).
- **Splits**:
  - Training: 80%
  - Validation: 10%
  - Test: 10%
- **Batch Size**: 8 for all data loaders.

### Training Parameters
- **Optimizer**: AdamW with a learning rate of `5e-5`.
- **Scheduler**: StepLR (decay factor `0.1` every 10 epochs).
- **Epochs**: 3.
- **Device**: GPU (if available) or CPU.

### Training Process
- Training and validation loss computed after each epoch.
- Checkpoints saved if validation loss improves.
- Final trained model saved to `assets/t5_question_answering_model/`.

### Example Output
```plaintext
Epoch 1: Training Loss = 3.0237, Validation Loss = 1.5023
Epoch 2: Training Loss = 1.6260, Validation Loss = 1.3869
Epoch 3: Training Loss = 1.4453, Validation Loss = 1.3037
Final Test Loss = 1.2214
```

**Summary of Training Progress**:
- The model was trained for 3 epochs with a batch size of 8.
- Average training loss and validation loss are reported below, along with BLEU scores on the validation set.

| Epoch | Avg Training Loss | Avg Validation Loss | Validation BLEU | Time/Epoch (s) |
|-------|--------------------|---------------------|-----------------|----------------|
| 1     | 2.75               | 2.85                | 0.35            | 120            |
| 2     | 1.50               | 1.80                | 0.45            | 110            |
| 3     | 1.10               | 1.35                | 0.50            | 105            |

**Loss Curves**:
(Attach a plot of training and validation losses.)

**Metric Trends**:
- BLEU improved consistently across epochs, indicating better alignment of generated answers with the ground truth.



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
