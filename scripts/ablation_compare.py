import argparse
import json
import os


def load_metrics(run_dir: str) -> dict:
    path = os.path.join(run_dir, "eval_metrics.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args()

    for run in args.runs:
        metrics = load_metrics(run)
        print(f"{run} -> {metrics}")


if __name__ == "__main__":
    main()
