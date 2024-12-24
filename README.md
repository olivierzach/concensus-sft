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
Epoch 1 Training Complete - Average Loss: 3.1219

Evaluating on validation set...
Epoch 1 Validation Loss: 0.5330

Validation loss improved. Saving checkpoint...
Checkpoint saved: assets/checkpoints/checkpoint_epoch_1.pt

Epoch 2 Training Complete - Average Loss: 0.9136

Evaluating on validation set...
Epoch 2 Validation Loss: 0.5267

Validation loss improved. Saving checkpoint...
Checkpoint saved: assets/checkpoints/checkpoint_epoch_2.pt

Epoch 3 Training Complete - Average Loss: 0.6704

Evaluating on validation set...
Epoch 3 Validation Loss: 0.5031

Validation loss improved. Saving checkpoint...
Checkpoint saved: assets/checkpoints/checkpoint_epoch_3.pt

Epoch 4 Training Complete - Average Loss: 0.6028

Evaluating on validation set...
Epoch 4 Validation Loss: 0.4719

Validation loss improved. Saving checkpoint...
Checkpoint saved: assets/checkpoints/checkpoint_epoch_4.pt

Epoch 5 Training Complete - Average Loss: 0.5714

Evaluating on validation set...
Epoch 5 Validation Loss: 0.4496

Validation loss improved. Saving checkpoint...
Checkpoint saved: assets/checkpoints/checkpoint_epoch_5.pt

Epoch 6 Training Complete - Average Loss: 0.5373

Evaluating on validation set...
Epoch 6 Validation Loss: 0.4328

Validation loss improved. Saving checkpoint...
Checkpoint saved: assets/checkpoints/checkpoint_epoch_6.pt

Epoch 7 Training Complete - Average Loss: 0.5175

Evaluating on validation set...
Epoch 7 Validation Loss: 0.4176

Validation loss improved. Saving checkpoint...
Checkpoint saved: assets/checkpoints/checkpoint_epoch_7.pt

Epoch 8 Training Complete - Average Loss: 0.4945

Evaluating on validation set...
Epoch 8 Validation Loss: 0.4029

Validation loss improved. Saving checkpoint...
Checkpoint saved: assets/checkpoints/checkpoint_epoch_8.pt

Epoch 9 Training Complete - Average Loss: 0.4743

Evaluating on validation set...
Epoch 9 Validation Loss: 0.3867

Validation loss improved. Saving checkpoint...
Checkpoint saved: assets/checkpoints/checkpoint_epoch_9.pt

Epoch 10 Training Complete - Average Loss: 0.4498

Evaluating on validation set...
Epoch 10 Validation Loss: 0.3657

Validation loss improved. Saving checkpoint...
Checkpoint saved: assets/checkpoints/checkpoint_epoch_10.pt
Final model saved to assets/t5_question_answering_model
Test Loss: 0.3859
```

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

## Why These Metrics?

| Metric         | Purpose                                   | Justification                                      |
|----------------|-------------------------------------------|--------------------------------------------------|
| **BLEU**       | Measures n-gram precision.               | Ensures answers contain key terms from references. |
| **ROUGE**      | Measures word/phrase overlap.            | Evaluates content coverage for longer answers.    |
| **Exact Match**| Checks strict correctness.               | Useful for tasks requiring exact technical phrasing. |
| **BERTScore**  | Measures semantic similarity.            | Captures paraphrases and high-level meaning.       |

---

## How to Run

```bash
python inference.py
```

```bash
python models/evaluation.py
```
