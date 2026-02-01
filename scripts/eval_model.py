import argparse
import json
import os

import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from consensus_sft.data import load_raw_dataframe, split_dataset, tokenize_splits
from consensus_sft.metrics import compute_text_metrics
from consensus_sft.utils import load_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["project"]["seed"])

    df = load_raw_dataframe(
        config["data"]["train_path"],
        config["data"]["text_columns"],
        config["data"]["label_column"],
    )
    splits = split_dataset(
        df,
        config["project"]["seed"],
        config["data"]["split"]["train"],
        config["data"]["split"]["val"],
        config["data"]["split"]["test"],
    )

    model_path = args.checkpoint or os.path.join(
        config["project"]["output_dir"],
        config["project"]["run_name"],
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    tokenized = tokenize_splits(
        splits,
        tokenizer,
        config["data"]["max_source_length"],
        config["data"]["max_target_length"],
    )

    test = tokenized["test"]
    outputs = model.generate(
        input_ids=test["input_ids"],
        attention_mask=test["attention_mask"],
        max_length=config["train"]["generation_max_length"],
        num_beams=config.get("generation", {}).get("num_beams", 4),
    )

    preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    labels = np.where(np.array(test["labels"]) != -100, test["labels"], tokenizer.pad_token_id)
    refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

    metrics = compute_text_metrics(preds, refs)
    with open(os.path.join(model_path, "eval_metrics_manual.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Evaluation complete:", metrics)


if __name__ == "__main__":
    main()
