import argparse
import os

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output", default="models/onnx")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)

    # Placeholder for ONNX export; for full export, use optimum onnxruntime.
    tokenizer.save_pretrained(args.output)
    model.save_pretrained(args.output)
    print(f"Saved model and tokenizer to {args.output} (use optimum for ONNX export).")


if __name__ == "__main__":
    main()
