# Failure Modes & Fixes

This project went through several failure modes before reaching a stable, high‑quality run. Below is a concise map of what broke, how it showed up, and the fix that worked.

## 1) Model repeats text / outputs are incoherent
- Symptom: repetitive or degenerate generations; outputs look like the input; BLEU/ROUGE near zero.
- Root cause: targets were noisy or misaligned with inputs; the model had no consistent supervision signal.
- Fix: replace targets with an **oracle extract** from the abstract, and drop low‑overlap rows.
- Code: `data/process_data_sft_clean.py` → produces `data/processed/training_data_sft_clean_end.csv`.

## 2) Training loss drops, eval gets worse
- Symptom: train loss steadily decreases, eval loss/ROUGE degrades after a few epochs.
- Root cause: rapid overfitting on a small dataset.
- Fix: early stopping on **eval ROUGE‑L** with higher patience; keep `load_best_model_at_end: true`.
- Config: `configs/consensus/clean_fit_end_es_rouge_long.yaml`.

## 3) Metrics fluctuate wildly across runs
- Symptom: large variance in BLEU/ROUGE between runs.
- Root cause: small dataset + unstable targets + evaluation noise.
- Fix: consistent oracle targets, reduced eval noise (less aggressive early stopping), and fixed seeds.

## 4) Outputs end abruptly / incomplete sentences
- Symptom: generations cut mid‑sentence even with long max length.
- Root cause: EOS token not aligned to the task; model ends early.
- Fix: add `[END]` as a special token and set it as EOS; ensure training targets end with `[END]`.
- Config: `configs/consensus/clean_fit_end_es_rouge_long.yaml`.

## 5) MPS memory issues / crashes
- Symptom: crashes or memory errors on Apple Silicon.
- Root cause: MPS allocator pressure + large intermediate activations.
- Fix: gradient checkpointing + periodic cache clears + lower MPS watermark.
- Settings: `gradient_checkpointing: true`, `torch_empty_cache_steps: 50`, `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`.

## 6) Good eval_loss, mediocre ROUGE
- Symptom: loss improves, but ROUGE stagnates or drops.
- Root cause: token‑level loss is not aligned with generation quality.
- Fix: early stopping and checkpoint selection on `eval_rougeL`, not eval loss.

## 7) Training gets slower over time
- Symptom: step time increases or eval dominates runtime.
- Root cause: frequent eval with beam search on a small device.
- Fix: reduce eval frequency (`eval_steps`), keep beams modest, and avoid large `max_length` unless needed.

## 8) Inference uses wrong model artifacts
- Symptom: inference results look like old/weak runs.
- Root cause: inference script points to legacy model path.
- Fix: use `scripts/infer.py` with the best run config, or point directly to the best checkpoint.

