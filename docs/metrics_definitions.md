# Metrics Definitions

This project reports BLEU and ROUGE (1/2/L). Below are the standard definitions used in practice and how they map to the codebase.

## BLEU (Papineni et al., 2002)
BLEU is a geometric mean of clipped n‑gram precisions with a brevity penalty:

- **Clipped n‑gram precision** for n in {1..N}:
  - `p_n = (sum over n-grams min(count_candidate, count_reference)) / (sum over n-grams count_candidate)`
- **Brevity penalty (BP)**:
  - `BP = 1` if `c > r`, else `exp(1 - r/c)`
  - `c = total candidate length`, `r = total reference length`
- **BLEU**:
  - `BLEU = BP * exp( (1/N) * sum_{n=1..N} log(p_n) )`

## ROUGE‑1 / ROUGE‑2 (Lin, 2004)
ROUGE‑N measures n‑gram overlap, typically reported as F1:

- **Recall**: overlap / reference n‑grams
- **Precision**: overlap / candidate n‑grams
- **F1**: `2 * (P * R) / (P + R)`

ROUGE‑1 uses unigrams, ROUGE‑2 uses bigrams.

## ROUGE‑L (Longest Common Subsequence)
ROUGE‑L is based on LCS between candidate and reference:

- `LCS = length of longest common subsequence`
- **Recall**: `R = LCS / len(reference)`
- **Precision**: `P = LCS / len(candidate)`
- **F1**: `ROUGE‑L = (1 + beta^2) * P * R / (R + beta^2 * P)`

The common setting is `beta = 1` (balanced F1).

## How We Compute Metrics
- **Primary**: Hugging Face `evaluate` package (`rouge`, `bleu`), with ROUGE stemming enabled.
- **Fallback**: `rouge_score` for ROUGE and NLTK `corpus_bleu` for BLEU.
- Code: `src/consensus_sft/metrics.py`

