import argparse
import csv

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="reports/pareto.png")
    args = parser.parse_args()

    runs, latency, rouge = [], [], []
    with open(args.input, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            runs.append(row.get("run", ""))
            latency.append(float(row.get("latency", 0.0)))
            rouge.append(float(row.get("rougeL", 0.0)))

    plt.figure(figsize=(6, 4))
    plt.scatter(latency, rouge)
    for i, name in enumerate(runs):
        plt.annotate(name, (latency[i], rouge[i]))
    plt.xlabel("Latency (s)")
    plt.ylabel("ROUGE-L")
    plt.title("Latency vs Accuracy")
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
