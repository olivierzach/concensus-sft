import argparse
import os

import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if "target_text" not in df.columns:
        raise ValueError("Expected a target_text column in the input CSV.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    if args.device != "cpu":
        model = model.to(args.device)

    paraphrases = []
    for i in range(0, len(df), args.batch_size):
        batch = df["target_text"].iloc[i : i + args.batch_size].tolist()
        prompts = [f"Paraphrase: {t}" for t in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        if args.device != "cpu":
            inputs = {k: v.to(args.device) for k, v in inputs.items()}

        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=4,
        )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        paraphrases.extend(decoded)

    df = df.copy()
    df["target_text_paraphrase"] = paraphrases

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved paraphrased dataset: {args.output_csv}")


if __name__ == "__main__":
    main()
