import argparse
import csv
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", default="reports/metrics_log.csv")
    parser.add_argument("--run_name", required=True)
    args = parser.parse_args()

    with open(args.metrics, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    file_exists = os.path.exists(args.output)
    with open(args.output, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["run", *metrics.keys()])
        writer.writerow([args.run_name, *metrics.values()])

    print(f"Logged metrics to {args.output}")


if __name__ == "__main__":
    main()
