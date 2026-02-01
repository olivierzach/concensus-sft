import argparse
import os

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--sample_size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    sample = df.sample(n=min(args.sample_size, len(df)), random_state=args.seed).copy()
    sample["quality_label"] = ""
    sample["notes"] = ""

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    sample.to_csv(args.output_csv, index=False)
    print(f"Saved review sample: {args.output_csv}")


if __name__ == "__main__":
    main()
