# Learning Summary

## Key Points
- Small datasets lead to fast overfitting; metrics plateau early.
- ROUGE/BLEU can diverge from token-level loss trends.
- Data quality and target design drive most gains.
- Latency is dominated by decoding strategy (beams vs greedy).

## Takeaways
- If metrics stall, examine target consistency first.
- Use smaller max lengths to stabilize training on constrained hardware.
- Always compare latency and quality, not just one metric.
- Keep raw data unchanged; separate ETL from training.
