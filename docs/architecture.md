# Model & Training Architecture

This project fine‑tunes a T5‑style encoder‑decoder model using supervised fine‑tuning (SFT). This doc gives a quick mental model of how it works.

## T5 in One Page
- **Text‑to‑text**: everything is cast as “input text → output text.”
- **Encoder‑decoder**:
  - **Encoder** reads the input sequence into hidden states.
  - **Decoder** autoregressively generates the output sequence, attending to the encoder.
- **Pretraining**: T5 is pretrained with a **span‑corruption** objective (mask spans, predict them).

## SFT in This Repo
- **Input format**: `Question: ... Context: ...`
- **Target**: a short answer / summary derived from the abstract.
- **Loss**: token‑level cross‑entropy over the target sequence.
- **Decoding**: beam search with repetition penalties; `[END]` is used as EOS to end generations cleanly.

## Why Oracle Targets Help
Early runs used noisy targets. We switched to **oracle extractive targets**:
- Extract the most relevant sentence(s) from the abstract.
- Drop low‑overlap rows.
This creates consistent supervision and stabilizes learning.

## Practical Details
- **Early stopping**: monitored on `eval_rougeL` to align with generation quality.
- **Apple Silicon**: MPS-friendly settings (gradient checkpointing, cache clears).
- **Best run config**: `configs/consensus/clean_fit_end_es_rouge_long.yaml`

