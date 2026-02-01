import argparse
import time

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=64)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)

    inputs = tokenizer([args.text], return_tensors="pt", padding=True, truncation=True)

    times = []
    for _ in range(args.runs):
        start = time.perf_counter()
        _ = model.generate(**inputs, max_length=args.max_length)
        times.append(time.perf_counter() - start)

    avg = sum(times) / len(times)
    print(f"Average latency: {avg:.4f}s over {args.runs} runs")


if __name__ == "__main__":
    main()
