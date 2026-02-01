# ETL Notes

## Source (Raw)
- `data/input_datasets/training_data.csv` is kept untouched for lineage.

## Processed (SFT-ready)
- `data/processed/training_data_sft.csv` is derived via `data/process_data_sft.py`.
- `data/processed/training_data_oracle.csv` uses oracle extractive targets for higher signal.

### Transformations
- Normalize whitespace and quotes in labels.
- Add instruction-style prompt.
- Trim abstracts to the first N sentences.
- Build oracle targets by selecting top abstract sentences via ROUGE-L vs (query + label).
- Optional bullet-style targets from oracle sentences.
- Filter rows with low label/abstract token overlap.
- Sample a review set for human spot checks.
- Optional paraphrase augmentation using a local seq2seq model.

### Rationale
- Prompts improve instruction following.
- Shorter, consistent context reduces noise.
- Overlap filter removes low-signal pairs.

To reproduce (label targets):
```bash
python data/process_data_sft.py \
  --input_csv data/input_datasets/training_data.csv \
  --output_csv data/processed/training_data_sft.csv \
  --target_style label \
  --min_overlap 0.45 \
  --max_label_tokens 40 \
  --min_label_tokens 5 \
  --max_abstract_sentences 8
```

To reproduce (oracle bullets):
```bash
python data/process_data_sft.py \
  --input_csv data/input_datasets/training_data.csv \
  --output_csv data/processed/training_data_oracle.csv \
  --target_style bullets \
  --oracle_reference query_label \
  --oracle_sentences 3 \
  --min_sentence_tokens 5 \
  --min_overlap 0.45 \
  --max_label_tokens 40 \
  --min_label_tokens 5 \
  --max_abstract_sentences 8
```

To create a review sample:
```bash
python data/sample_for_review.py \
  --input_csv data/processed/training_data_oracle.csv \
  --output_csv data/processed/review_sample.csv \
  --sample_size 50
```

To add local paraphrases (requires a local or cached model):
```bash
python data/augment_paraphrases.py \
  --input_csv data/processed/training_data_oracle.csv \
  --output_csv data/processed/training_data_oracle_paraphrase.csv \
  --model_name_or_path t5-small \
  --device cpu
```
