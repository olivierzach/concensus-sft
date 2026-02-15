import argparse
import csv
import json
import os
import sys
from typing import Dict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCITLDR_CACHE = os.path.join(REPO_ROOT, "data", "scitldr", ".cache")
os.makedirs(SCITLDR_CACHE, exist_ok=True)
os.environ.setdefault("HF_HOME", SCITLDR_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(SCITLDR_CACHE, "transformers"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)

from consensus_sft.metrics import compute_text_metrics
from consensus_sft.utils import ensure_dir, load_config, set_seed


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


def load_splits(config: dict) -> DatasetDict:
    train_df = pd.read_csv(config["data"]["train_path"])
    val_df = pd.read_csv(config["data"]["val_path"])
    test_df = pd.read_csv(config["data"]["test_path"])

    return DatasetDict(
        train=Dataset.from_pandas(train_df, preserve_index=False),
        validation=Dataset.from_pandas(val_df, preserve_index=False),
        test=Dataset.from_pandas(test_df, preserve_index=False),
    )


def tokenize_splits(dataset_dict: DatasetDict, tokenizer, max_source_length: int, max_target_length: int) -> DatasetDict:
    def tokenize_batch(batch):
        model_inputs = tokenizer(
            batch["input_text"],
            max_length=max_source_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target_text"],
            max_length=max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset_dict.map(tokenize_batch, batched=True, remove_columns=["input_text", "target_text"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["project"]["seed"])

    output_dir = os.path.join(config["project"]["output_dir"], config["project"]["run_name"])
    ensure_dir(output_dir)

    splits = load_splits(config)

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["model_name"])
    tokenized = tokenize_splits(
        splits,
        tokenizer,
        config["data"]["max_source_length"],
        config["data"]["max_target_length"],
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(config["model"]["model_name"])
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["train"]["num_train_epochs"],
        per_device_train_batch_size=config["train"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["train"]["per_device_eval_batch_size"],
        learning_rate=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"],
        warmup_ratio=config["train"]["warmup_ratio"],
        gradient_accumulation_steps=config["train"]["gradient_accumulation_steps"],
        evaluation_strategy=config["train"]["eval_strategy"],
        save_strategy=config["train"]["save_strategy"],
        save_total_limit=config["train"]["save_total_limit"],
        logging_steps=config["train"]["logging_steps"],
        predict_with_generate=config["train"]["predict_with_generate"],
        generation_max_length=config["train"]["generation_max_length"],
        fp16=config["train"]["fp16"],
        load_best_model_at_end=config["train"]["load_best_model_at_end"],
        metric_for_best_model=config["train"]["metric_for_best_model"],
        greater_is_better=config["train"]["greater_is_better"],
        report_to="none",
    )

    callbacks = []
    if config.get("early_stopping", {}).get("enabled"):
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=config["early_stopping"].get("patience", 2)))

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
        callbacks=callbacks,
    )

    generation = config.get("generation", {})
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

    print(f"Training complete. Model saved to {output_dir}.")


if __name__ == "__main__":
    main()
