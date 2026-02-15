import argparse
import json
import os
import csv
import shutil

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from consensus_sft.data import load_raw_dataframe, split_dataset
from consensus_sft.metrics import compute_text_metrics
from consensus_sft.utils import load_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--export_best", action="store_true")
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
    test = splits["test"]
    input_texts = list(test["input_text"])
    reference_texts = list(test["target_text"])
    inputs = tokenizer(
        input_texts,
        max_length=config["data"]["max_source_length"],
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=config["train"]["generation_max_length"],
        num_beams=config.get("generation", {}).get("num_beams", 4),
    )

    preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    refs = reference_texts

    metrics = compute_text_metrics(preds, refs)
    with open(os.path.join(model_path, "eval_metrics_manual.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if args.save_predictions:
        with open(os.path.join(model_path, "eval_predictions.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["predicted", "reference"])
            writer.writerows(zip(preds, refs))

    if args.export_best and args.checkpoint:
        export_dir = os.path.join(
            config["project"]["output_dir"],
            config["project"]["run_name"],
            "best_checkpoint",
        )
        if os.path.exists(export_dir):
            shutil.rmtree(export_dir)
        os.makedirs(export_dir, exist_ok=True)
        tokenizer.save_pretrained(export_dir)
        model.save_pretrained(export_dir)

    print("Evaluation complete:", metrics)


if __name__ == "__main__":
    main()
