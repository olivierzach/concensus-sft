import argparse
import csv
import json
import os
import time
from typing import Dict

import numpy as np
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
)

from consensus_sft.data import load_raw_dataframe, split_dataset, tokenize_splits
from consensus_sft.metrics import compute_text_metrics
from consensus_sft.utils import ensure_dir, get_device, load_config, set_seed


def build_compute_metrics(tokenizer):
    def compute_metrics(eval_pred) -> Dict[str, float]:
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        if preds.ndim == 3:
            preds = np.argmax(preds, axis=-1)
        if not np.issubdtype(preds.dtype, np.integer):
            preds = preds.astype(np.int64)
        vocab_size = getattr(tokenizer, "vocab_size", None)
        if vocab_size:
            preds = np.clip(preds, 0, vocab_size - 1)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        return compute_text_metrics(decoded_preds, decoded_labels)

    return compute_metrics


class TrainingProfileCallback(TrainerCallback):
    def __init__(self, device: str) -> None:
        self.device = device
        self._start_time = None
        self._last_log_time = None
        self._last_log_step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self._start_time = time.perf_counter()
        self._last_log_time = self._start_time

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or state.global_step == 0:
            return
        now = time.perf_counter()
        elapsed = now - (self._last_log_time or now)
        steps = state.global_step - self._last_log_step
        if steps <= 0:
            return
        step_time = elapsed / steps
        logs["profile_step_time_sec"] = round(step_time, 4)
        logs["profile_steps_per_sec"] = round(1.0 / step_time, 4) if step_time > 0 else None
        if self.device == "mps" and torch.backends.mps.is_available():
            try:
                logs["mps_allocated_mb"] = round(torch.mps.current_allocated_memory() / 1024**2, 2)
            except Exception:
                pass
        self._last_log_step = state.global_step
        self._last_log_time = now


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config["project"]["seed"]
    set_seed(seed)

    output_dir = os.path.join(config["project"]["output_dir"], config["project"]["run_name"])
    ensure_dir(output_dir)

    df = load_raw_dataframe(
        config["data"]["train_path"],
        config["data"]["text_columns"],
        config["data"]["label_column"],
    )
    splits = split_dataset(
        df,
        seed,
        config["data"]["split"]["train"],
        config["data"]["split"]["val"],
        config["data"]["split"]["test"],
    )

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["model_name"])
    tokenized = tokenize_splits(
        splits,
        tokenizer,
        config["data"]["max_source_length"],
        config["data"]["max_target_length"],
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(config["model"]["model_name"])
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    eval_strategy = config["train"].get("eval_strategy", config["train"].get("evaluation_strategy", "epoch"))
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["train"]["num_train_epochs"],
        per_device_train_batch_size=config["train"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["train"]["per_device_eval_batch_size"],
        learning_rate=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"],
        warmup_ratio=config["train"]["warmup_ratio"],
        gradient_accumulation_steps=config["train"]["gradient_accumulation_steps"],
        eval_strategy=eval_strategy,
        save_strategy=config["train"]["save_strategy"],
        save_total_limit=config["train"]["save_total_limit"],
        logging_steps=config["train"]["logging_steps"],
        predict_with_generate=config["train"]["predict_with_generate"],
        generation_max_length=config["train"]["generation_max_length"],
        fp16=config["train"]["fp16"],
        gradient_checkpointing=config["train"].get("gradient_checkpointing", False),
        torch_empty_cache_steps=config["train"].get("torch_empty_cache_steps", None),
        use_cpu=config["train"].get("use_cpu", False),
        load_best_model_at_end=config["train"].get("load_best_model_at_end", False),
        metric_for_best_model=config["train"].get("metric_for_best_model", None),
        greater_is_better=config["train"].get("greater_is_better", None),
        report_to="none",
    )

    if config["train"].get("gradient_checkpointing", False):
        model.config.use_cache = False

    generation = config.get("generation", {})
    callbacks = []
    if config.get("early_stopping", {}).get("enabled"):
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=config["early_stopping"].get("patience", 2)))
    if config.get("profiling", {}).get("enabled"):
        callbacks.append(TrainingProfileCallback(device=get_device()))

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
        callbacks=callbacks,
    )

    if generation:
        trainer.model.generation_config.update(**generation)

    trainer.train()

    metrics = trainer.evaluate()
    with open(os.path.join(output_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    test_predictions = trainer.predict(tokenized["test"], max_length=config["train"]["generation_max_length"])
    preds = test_predictions.predictions
    if preds.ndim == 3:
        preds = np.argmax(preds, axis=-1)
    if not np.issubdtype(preds.dtype, np.integer):
        preds = preds.astype(np.int64)
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size:
        preds = np.clip(preds, 0, vocab_size - 1)
    preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(test_predictions.label_ids != -100, test_predictions.label_ids, tokenizer.pad_token_id)
    refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

    predictions_path = os.path.join(output_dir, "test_predictions.csv")
    with open(predictions_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["predicted", "reference"])
        writer.writerows(zip(preds, refs))

    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)

    device = get_device()
    print(f"Training complete. Model saved to {output_dir} (device: {device}).")


if __name__ == "__main__":
    main()
