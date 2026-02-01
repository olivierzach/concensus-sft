import argparse
import csv
import os

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from consensus_sft.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/predictions.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    model_path = os.path.join(config["project"]["output_dir"], config["project"]["run_name"])

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    with open(args.input, "r", encoding="utf-8") as f:
        rows = [line.strip() for line in f if line.strip()]

    outputs = model.generate(
        **tokenizer(rows, return_tensors="pt", padding=True, truncation=True),
        max_length=config["train"]["generation_max_length"],
        num_beams=config.get("generation", {}).get("num_beams", 4),
    )

    preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["input", "prediction"])
        writer.writerows(zip(rows, preds))

    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
