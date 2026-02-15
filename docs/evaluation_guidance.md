# Evaluation Guidance

## Automatic Metrics
- **ROUGE-1/2/L**: overlap-based; good for surface similarity.
- **BLEU**: precision-heavy; harsher for abstractive summaries.
- **BERTScore/BLEURT**: semantic similarity; better for paraphrases.

## Faithfulness vs Correctness
- **Correctness**: is the output factually accurate.
- **Faithfulness**: is the output supported by the given context.

## Suggested Protocol
1) Report ROUGE + BLEU for baseline comparability.
2) Add semantic metric for robustness.
3) Use NLI-based entailment on a sample for faithfulness.
4) Include a short human evaluation pass for qualitative sanity.

## Metric Definitions
See `docs/metrics_definitions.md` for formulas and computation details.
