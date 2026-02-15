# Learning Summary

## Key Points
- Small datasets lead to fast overfitting; metrics plateau early.
- ROUGE/BLEU can diverge from token-level loss trends.
- Data quality and target design drive most gains.
- Latency is dominated by decoding strategy (beams vs greedy).

## Best Run Summary
- Run: `configs/consensus/clean_fit_end_es_rouge_long.yaml`
- Output: `outputs/consensus_clean/flan_t5_small_clean_end_es_rouge_long/`
- Best checkpoint: `outputs/consensus_clean/flan_t5_small_clean_end_es_rouge_long/best_checkpoint/`
- Metrics: BLEU 0.6237, ROUGE-1 0.7143, ROUGE-2 0.6352, ROUGE-L 0.6637
- Early stopping: monitored `eval_rougeL`, patience 25, `load_best_model_at_end: true`
- Best checkpoint: step 10000 (see `best_checkpoint/`)
- Best eval_rougeL observed in logs: 0.6645 at epoch ~37.34 (step 8700)
- Training duration: not recorded in the saved state; estimate requires log capture during run
- Key data change: targets replaced with oracle sentence extracts from the abstract; low-overlap rows dropped
- Objective alignment: optimized for generation quality (ROUGE-L), not just token-level loss
- Decoding setup: `[END]` token used as EOS to allow complete outputs; beam search with repetition control
- Stability settings: gradient checkpointing, MPS high watermark disabled, periodic cache clears

## Takeaways
- If metrics stall, examine target consistency first.
- Use smaller max lengths to stabilize training on constrained hardware.
- Always compare latency and quality, not just one metric.
- Keep raw data unchanged; separate ETL from training.

## Learning Summary Guide (Quick Return)
- Start with data: ensure input/target alignment before tuning models.
- Pick the metric you care about, then early stop on that metric.
- Expect overfitting on small datasets; checkpoint selection matters.
- Decoding choices (beams, length, EOS) directly shape output quality.
- Prefer reproducible configs and write down what changed each run.
