# Consensus SFT Take-Home

A clean, reproducible SFT baseline for scientific question answering with a focus on clarity, evaluation rigor, and latency/accuracy tradeoffs.

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Build processed dataset
python data/process_data_sft.py \
  --input_csv data/input_datasets/training_data.csv \
  --output_csv data/processed/training_data_sft.csv \
  --target_style label

# Train a small MPS-friendly model
python scripts/train.py --config configs/small_high_accuracy_mps.yaml
```

## Project Layout
- `src/consensus_sft/`: core data + metrics utilities
- `scripts/`: training/eval/inference/benchmark helpers
- `configs/`: reproducible configs
- `docs/`: learning materials and evaluation guidance
- `reports/`: model card + templates
- `data/`: raw + processed datasets

## Key Configs
- `configs/default.yaml`: baseline training
- `configs/low_latency.yaml`: faster decoding + shorter lengths
- `configs/small_high_accuracy_mps.yaml`: best small-model quality run
- `configs/high_accuracy.yaml`: larger model (CPU)

## Evaluation
```bash
python scripts/eval_model.py --config configs/small_high_accuracy_mps.yaml
python scripts/benchmark_inference.py --model_path outputs/flan_t5_small_high_accuracy_mps \
  --text "Question: ... Context: ..." --runs 20
```

## Learning Path
See:
- `docs/learning_path.md`
- `docs/learning_summary.md`
- `docs/evaluation_guidance.md`

## Notes
- Raw data is never mutated. All transformations go through `data/process_data_sft.py`.
- Outputs are intentionally not tracked in git.
