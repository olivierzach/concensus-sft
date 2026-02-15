import argparse
import csv
import json
import os
import random
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

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


def load_splits(path: str, seed: int, train_frac: float, val_frac: float, test_frac: float) -> DatasetDict:
    df = pd.read_csv(path)
    if "input_text" not in df.columns or "target_text" not in df.columns:
        raise ValueError("Processed CSV must contain input_text and target_text columns.")
    df = df[["input_text", "target_text"]].copy()
    dataset = Dataset.from_pandas(df, preserve_index=False)
    temp_split = dataset.train_test_split(test_size=val_frac + test_frac, seed=seed)
    val_ratio = val_frac / (val_frac + test_frac)
    val_test = temp_split["test"].train_test_split(test_size=1 - val_ratio, seed=seed)
    return DatasetDict(
        train=temp_split["train"],
        validation=val_test["train"],
        test=val_test["test"],
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


class SanityCheckCallback(TrainerCallback):
    def __init__(self, tokenizer, samples: List[Dict[str, str]], output_dir: str) -> None:
        self.tokenizer = tokenizer
        self.samples = samples
        self.output_dir = output_dir

    def on_evaluate(self, args, state, control, **kwargs):
        if not self.samples:
            return
        model = kwargs.get("model")
        if model is None:
            return
        device = next(model.parameters()).device
        rows = []
        for sample in self.samples:
            prompt = sample["input_text"]
            target = sample["target_text"]
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.generation_max_length,
            ).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.generation_max_length,
                    num_beams=4,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.1,
                )
            pred = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            rows.append(
                {
                    "step": state.global_step,
                    "input_text": prompt,
                    "target_text": target,
                    "prediction": pred,
                }
            )
        out_path = os.path.join(self.output_dir, f"sanity_predictions_step_{state.global_step}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume_from_checkpoint", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config["project"]["seed"]
    set_seed(seed)

    output_dir = os.path.join(config["project"]["output_dir"], config["project"]["run_name"])
    ensure_dir(output_dir)

    splits = load_splits(
        config["data"]["processed_path"],
        seed,
        config["data"]["split"]["train"],
        config["data"]["split"]["val"],
        config["data"]["split"]["test"],
    )

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["model_name"], use_fast=False)
    if config.get("special_tokens"):
        added = tokenizer.add_special_tokens(config["special_tokens"])
        if added:
            print(f"Added {added} special tokens.")
    tokenized = tokenize_splits(
        splits,
        tokenizer,
        config["data"]["max_source_length"],
        config["data"]["max_target_length"],
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(config["model"]["model_name"])
    if config.get("special_tokens"):
        model.resize_token_embeddings(len(tokenizer))
    if config.get("end_token"):
        end_token_id = tokenizer.convert_tokens_to_ids(config["end_token"])
        if end_token_id is not None and end_token_id != tokenizer.unk_token_id:
            model.generation_config.eos_token_id = end_token_id
            model.config.eos_token_id = end_token_id
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

    if config["train"].get("gradient_checkpointing", False):
        model.config.use_cache = False

    samples = []
    if config.get("sanity", {}).get("enabled"):
        sample_size = config["sanity"].get("num_samples", 3)
        val_df = splits["validation"].to_pandas()
        rng = random.Random(seed)
        idxs = rng.sample(range(len(val_df)), k=min(sample_size, len(val_df)))
        for idx in idxs:
            row = val_df.iloc[idx]
            samples.append({"input_text": row["input_text"], "target_text": row["target_text"]})

    callbacks: List[TrainerCallback] = []
    if samples:
        callbacks.append(SanityCheckCallback(tokenizer, samples, output_dir))
    if config.get("early_stopping", {}).get("enabled"):
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config["early_stopping"].get("patience", 2)
            )
        )

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

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

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
