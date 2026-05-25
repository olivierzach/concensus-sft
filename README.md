# Supervised Fine-Tuning for Scientific QA

A clean, reproducible supervised fine-tuning (SFT) baseline for scientific question answering (QA) with a focus on clarity, evaluation rigor, and latency/accuracy tradeoffs.

The task is based on Consensus-style scientific QA: given a user question and a paper abstract, generate a short answer grounded in the paper context.

## Abstract
We build a supervised fine-tuning (SFT) pipeline for scientific question answering. Early runs failed due to noisy targets and misalignment, producing repetitive outputs. We introduce an oracle extractive target construction and filtering strategy that aligns supervision with the input context. With early stopping on ROUGE‑L and MPS‑friendly training settings, the final model reaches ROUGE‑1 > 0.70 and ROUGE‑L ~0.664 while remaining feasible on a Mac mini.

## Problem Definition
Given a **question** and a **paper abstract**, generate a short, faithful answer. This is a small‑data regime with noisy labels, so target design and evaluation alignment dominate performance.

## Data & Preprocessing
Raw inputs include `query`, `context`, and a noisy `label`. The clean pipeline:
1) Builds an **oracle target** by selecting the most relevant sentence(s) from the abstract.
2) Drops rows with low overlap between input and target.
3) Appends `[END]` to targets and uses it as EOS at decode time.

See `data/process_data_sft_clean.py` for implementation.

## Method (Model + Objective)
- **Model**: FLAN‑T5 small (encoder‑decoder, text‑to‑text)
- **Input**: `Question: ... Context: ...`
- **Objective**: token‑level cross‑entropy on oracle targets
- **Decoding**: beam search + repetition control + `[END]` as EOS

## Training Details
Best‑run settings (see `configs/consensus/clean_fit_end_es_rouge_long.yaml`):
- Batch size 4, LR 1.5e‑5, weight decay 0.01
- Early stopping on `eval_rougeL` (patience 25)
- Gradient checkpointing + MPS cache controls

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

## Evaluation Protocol
- **Primary**: ROUGE‑L (summary quality, sequence coherence)
- **Secondary**: ROUGE‑1/2, BLEU
- **Selection**: early stopping on ROUGE‑L to align with generation quality
- **Metric formulas**: `docs/metrics_definitions.md`

## Results
Best run (oracle + early stopping on ROUGE‑L):
- BLEU **0.6237**
- ROUGE‑1 **0.7143**
- ROUGE‑2 **0.6352**
- ROUGE‑L **0.6637**

Iteration trajectory:
```
baseline (low latency):   BLEU 0.0587, ROUGE‑L 0.2094
clean (oracle) run:       BLEU 0.5564, ROUGE‑L 0.5865
long clean run:           BLEU 0.6032, ROUGE‑L 0.6538
best concensus-sft run:   BLEU 0.6237, ROUGE‑L 0.6637
```

## Project Layout
- `src/consensus_sft/`: core data + metrics utilities
- `scripts/`: training/eval/inference/benchmark helpers
- `configs/`: reproducible configs
- `docs/`: learning materials and evaluation guidance
- `reports/`: model card + templates
- `data/`: raw + processed datasets
- `data/source_materials/`: source brief, data dictionary, and supplied CSVs

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

## Latest Clean Run
- Config: `configs/consensus/clean_fit_end_es_rouge_long.yaml`
- Output: `outputs/consensus_clean/flan_t5_small_clean_end_es_rouge_long/`
- Best checkpoint: `outputs/consensus_clean/flan_t5_small_clean_end_es_rouge_long/best_checkpoint/`
- Metrics: BLEU 0.6237, ROUGE-1 0.7143, ROUGE-2 0.6352, ROUGE-L 0.6637
- Notes: targets are oracle sentence extracts; `[END]` token used as EOS for complete outputs; early stopping on `eval_rougeL` (patience 25)
- Callout: ROUGE-1 > 70% on the scientific QA task.

## Results Highlights (Iteration Journey)
- Baseline (low-latency) was weak: BLEU 0.0587, ROUGE-L 0.2094.
- First stable clean run (oracle targets) jumped to BLEU 0.5564, ROUGE-L 0.5865.
- Long clean run improved further: BLEU 0.6032, ROUGE-L 0.6538.
- Best concensus-sft run: BLEU 0.6237, ROUGE-L 0.6637.
- Net gain vs earliest baseline: ~10× BLEU and ~3× ROUGE-L, with coherent outputs.

## Engineering Notes
- **MPS stability**: gradient checkpointing + cache clears + low watermark.
- **Reproducibility**: configs are versioned; outputs are intentionally not tracked.
- **Failure modes**: see `docs/failure_modes.md`.

## Limitations & Next Steps
- Small dataset; metrics can be noisy across seeds.
- ROUGE/BLEU are imperfect for semantic correctness.
- Next steps: semantic metrics (BERTScore/BLEURT), NLI-based faithfulness checks, stronger data curation.

## Seed Sweep (Variance Check)
Run the 3‑seed sweep (no timeouts in your terminal):
```bash
cd concensus-sft
source .venv_scitldr/bin/activate
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python scripts/seed_sweep.py \
  --base_config configs/consensus/seed_sweep_base.yaml \
  --seeds 13,21,1337
```

Resume a seed if interrupted:
```bash
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python scripts/consensus/train_clean.py \
  --config /tmp/seed_sweep_13.yaml \
  --resume_from_checkpoint outputs/consensus_seed_sweep/flan_t5_small_clean_end_es_rouge_long_seed13/checkpoint-6000
```

## Learning Path
See:
- `docs/learning_path.md`
- `docs/learning_summary.md`
- `docs/sft_review_notes.md`
- `docs/evaluation_guidance.md`
- `docs/metrics_definitions.md`
- `docs/architecture.md`
- `docs/diagrams.md`

## Notes
- Raw data is never mutated. All transformations go through `data/process_data_sft.py`.
- Outputs are intentionally not tracked in git.
