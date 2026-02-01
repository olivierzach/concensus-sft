# Learning Path

## 1) Understand the Task
- Read the dataset schema in `data/input_datasets/`.
- Confirm what the label represents (short answer vs summary vs claim).

## 2) Data Pipeline
- Review `data/process_data_sft.py` and `docs/etl_notes.md`.
- Regenerate `data/processed/training_data_sft.csv`.
- Create a small review sample and annotate.

## 3) Baselines
- Run `configs/default.yaml` and record metrics.
- Compare `configs/low_latency.yaml` and `configs/small_high_accuracy_mps.yaml`.

## 4) Evaluation & Error Analysis
- Use `scripts/eval_model.py` and `scripts/faithfulness_nli.py`.
- Fill `reports/error_analysis_template.md` with failure cases.

## 5) Latency vs Accuracy
- Benchmark `scripts/benchmark_inference.py`.
- Plot Pareto with `scripts/plot_pareto.py`.

## 6) Stretch Goals
- Add semantic metrics (BERTScore / NLI).
- Try a larger model (flan-t5-base) on CPU.
- Export to ONNX for inference trials.
