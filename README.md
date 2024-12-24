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
- **Script**: `models/train.py`
- Metrics Used:
  1. **BLEU**: Measures n-gram overlap for lexical similarity.
  2. **ROUGE**: Evaluates unigram, bigram, and longest sequence overlap.

 
## Model Training Summary

### Model Details
- **Architecture**: `google/flan-t5-small`, Sequence-to-Sequence for open-ended text generation, instruction-tuned on flan dataset.
- **Pre-trained Model**: Hugging Face `oogle/flan-t5-small`, fine-tuned for scientific question answering.
- **Loss Function**: Cross-entropy (built into T5).

### Dataset
- **Input Data**: Preprocessed data (`data/output_datasets/preprocessed_data.pkl`).
- **Splits**:
  - Training: 80%
  - Validation: 10%
  - Test: 10%
- **Batch Size**: 12 for all data loaders.

### Tokenization
- concat question and abstract from training data into `question, context pairs`
- tokenize both inputs and labels (answers) with `max_length = 128` (very small window)

### Training Parameters
- **Optimizer**: AdamW with a learning rate of `1e-5`.
- **Scheduler**: StepLR (decay factor `0.1` every 10 epochs).
- **Epochs**: 5.
- **Device**: CPU (GPU poor)

### Training Process (really small training run to get some results)
- Training and validation loss computed after each epoch.
- Checkpoints saved if validation loss improves.
- Final trained model saved to `assets/t5_question_answering_model/`.

### Training Output
```plaintext
Epoch 1: Train Loss=2.9868, Validation Loss=2.5534
Checkpoint saved: assets/checkpoints/checkpoint_epoch_1.pt
  Checkpoint saved for Epoch 1

Epoch 2: Train Loss=2.8483, Validation Loss=2.4907
Checkpoint saved: assets/checkpoints/checkpoint_epoch_2.pt
  Checkpoint saved for Epoch 2

Epoch 3: Train Loss=2.7776, Validation Loss=2.4544
Checkpoint saved: assets/checkpoints/checkpoint_epoch_3.pt
  Checkpoint saved for Epoch 3

Epoch 4: Train Loss=2.7322, Validation Loss=2.4380
Checkpoint saved: assets/checkpoints/checkpoint_epoch_4.pt
  Checkpoint saved for Epoch 4

Epoch 5: Train Loss=2.6934, Validation Loss=2.4258
Checkpoint saved: assets/checkpoints/checkpoint_epoch_5.pt
  Checkpoint saved for Epoch 5
Final model and tokenizer saved to assets/t5_question_answering_model

```

---

## Evaluation Results

### Speed Test
- **Batch Size**: 20 inputs.
- **Elapsed Time**: average: 8.2 seconds, best: 7.2 seconds

Tried hard to tune-the knobs for better performance, but to no available. Several parameters are pushes towards the lightest inference, likely at the expense of performance. 
- 128 max sequence length
- greedy beam search (num_beams=1)
- used t5-small to fine-tune
- hacky process threads and batch size tuning
- no GPU, potentially this would hit the < 3 seconds limit
- attempted dynamic quantization, incompatible with t5 models

inference experiment examples: 

```plaintext
python /Users/zacholivier/Desktop/Projects/concensus-interview/inference.py \
    --data /Users/zacholivier/Desktop/Projects/concensus-interview/data/input_datasets/inference_data.csv \
    --model_dir /Users/zacholivier/Desktop/Projects/concensus-interview/models/assets/t5_question_answering_model \
    --mode multi \
    --batch_size 5 \
    --max_workers 4 \


Loading model and tokenizer...
Model and tokenizer loaded.
Reading data from /Users/zacholivier/Desktop/Projects/concensus-interview/data/input_datasets/inference_data.csv...
Data shape: (20, 7)
Running multi-threaded inference...
Processed batch of size 5 in 2.34 seconds.
Processed batch of size 5 in 4.81 seconds.
Processed batch of size 5 in 7.86 seconds.
Processed batch of size 5 in 7.86 seconds.
Predictions saved to inference_results.csv

python /Users/zacholivier/Desktop/Projects/concensus-interview/inference.py \
    --data /Users/zacholivier/Desktop/Projects/concensus-interview/data/input_datasets/inference_data.csv \
    --model_dir /Users/zacholivier/Desktop/Projects/concensus-interview/models/assets/t5_question_answering_model \
    --mode single \
    --batch_size 1 \
    --max_workers 12 \


Loading model and tokenizer...
Model and tokenizer loaded.
Reading data from /Users/zacholivier/Desktop/Projects/concensus-interview/data/input_datasets/inference_data.csv...
Data shape: (20, 7)
Running single-batch inference...
Single batch processed in 8.55 seconds.
Predictions saved to inference_results.csv

```
  

### Evaluation Metrics
- model is really under-trained model with minimal preprocessing does not do well in performance metrics, or eye-test

| Metric           | Value   |
|-------------------|---------|
| **Average BLEU**  | .081   |
| **Average ROUGE** | {'rouge1': 0.36019419947241854, 'rouge2': 0.16764773591750154, 'rougeL': 0.30468894899877236} |

---

## Why These Metrics?

| Metric         | Purpose                                   | Justification                                      |
|----------------|-------------------------------------------|--------------------------------------------------|
| **BLEU**       | Measures n-gram precision.               | Ensures answers contain key terms from references. |
| **ROUGE**      | Measures word/phrase overlap.            | Evaluates content coverage for longer answers.    |

---

## How to Run

```bash
python inference.py
```

```bash
python models/evaluation.py
```
