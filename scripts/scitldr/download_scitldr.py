import argparse
import os

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/scitldr/raw")
    parser.add_argument("--subset", default="Abstract")
    parser.add_argument("--max_samples", type=int, default=5000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ds = load_dataset("allenai/scitldr", args.subset)

    for split in ds.keys():
        data = ds[split]
        if args.max_samples and len(data) > args.max_samples:
            data = data.shuffle(seed=42).select(range(args.max_samples))
        out_path = os.path.join(args.output_dir, f"scitldr_{args.subset}_{split}.jsonl")
        data.to_json(out_path)
        print(f"Saved {split} to {out_path} (rows: {len(data)})")


if __name__ == "__main__":
    main()
