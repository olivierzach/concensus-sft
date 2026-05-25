# SFT Review Notes

## Modeling
- Why text-to-text framing works for SFT.
- Differences between extractive vs abstractive targets.
- Decoding tradeoffs: greedy vs beam vs sampling.

## Data
- How to keep raw data immutable and build ETL layers.
- Detecting label noise and low-signal examples.
- Train/val/test splits and leakage checks.

## Evaluation
- When ROUGE is misleading and how to complement it.
- Faithfulness vs correctness.
- Error analysis categories.

## Systems
- Latency drivers in seq2seq models.
- Memory bottlenecks (activation size, sequence length).
- Practical MPS/CPU tradeoffs on Mac hardware.
