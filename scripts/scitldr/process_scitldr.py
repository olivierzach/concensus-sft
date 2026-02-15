import argparse
import json
import os

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/scitldr/raw")
    parser.add_argument("--output_dir", default="data/scitldr/processed")
    parser.add_argument("--max_source_length", type=int, default=512)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for split in ["train", "validation", "test"]:
        in_path = os.path.join(args.input_dir, f"scitldr_Abstract_{split}.jsonl")
        rows = []
        with open(in_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                source = item.get("source", "")
                target = item.get("target", "")
                input_text = (
                    "Instruction: Summarize the abstract in 1-2 sentences. "
                    f"Abstract: {source}"
                )
                rows.append({"input_text": input_text, "target_text": target})

        df = pd.DataFrame(rows)
        out_path = os.path.join(args.output_dir, f"scitldr_{split}.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved {out_path} (rows: {len(df)})")


if __name__ == "__main__":
    main()
