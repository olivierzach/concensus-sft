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
- `configs/consensus/clean_fit.yaml`: cleaned QA-style pipeline (oracle targets)
- `configs/consensus/clean_fit_end.yaml`: cleaned QA-style + `[END]` stopping token

## Evaluation
```bash
python scripts/eval_model.py --config configs/small_high_accuracy_mps.yaml
python scripts/benchmark_inference.py --model_path outputs/flan_t5_small_high_accuracy_mps \
  --text "Question: ... Context: ..." --runs 20
```

## Inference (Best Run)
```bash
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python inference.py \
  --model_path outputs/consensus_clean/flan_t5_small_clean_end_es_rouge_long/best_checkpoint \
  --input_csv data/input_datasets/inference_data.csv
```

## Latest Clean Run (Consensus)
- Config: `configs/consensus/clean_fit_end_es_rouge_long.yaml`
- Output: `outputs/consensus_clean/flan_t5_small_clean_end_es_rouge_long/`
- Best checkpoint: `outputs/consensus_clean/flan_t5_small_clean_end_es_rouge_long/best_checkpoint/`
- Metrics: BLEU 0.6237, ROUGE-1 0.7143, ROUGE-2 0.6352, ROUGE-L 0.6637
- Notes: targets are oracle sentence extracts; `[END]` token used as EOS for complete outputs; early stopping on `eval_rougeL` (patience 25)
- Callout: ROUGE-1 > 70% on the consensus task.

## Results Highlights (Iteration Journey)
- Baseline (low-latency) was weak: BLEU 0.0587, ROUGE-L 0.2094.
- First stable clean run (oracle targets) jumped to BLEU 0.5564, ROUGE-L 0.5865.
- Long clean run improved further: BLEU 0.6032, ROUGE-L 0.6538.
- Best run (this repo): BLEU 0.6237, ROUGE-L 0.6637.
- Net gain vs earliest baseline: ~10× BLEU and ~3× ROUGE-L, with coherent outputs.

## Learning Path
See:
- `docs/learning_path.md`
- `docs/learning_summary.md`
- `docs/evaluation_guidance.md`
- `docs/metrics_definitions.md`
- `docs/architecture.md`
- `docs/diagrams.md`

## Notes
- Raw data is never mutated. All transformations go through `data/process_data_sft.py`.
- Outputs are intentionally not tracked in git.
