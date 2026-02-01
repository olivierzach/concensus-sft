import argparse
import csv

from transformers import pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default="reports/faithfulness_nli.csv")
    parser.add_argument("--model", default="roberta-large-mnli")
    args = parser.parse_args()

    classifier = pipeline("text-classification", model=args.model, return_all_scores=True)

    rows = []
    with open(args.predictions, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    results = []
    for row in rows:
        premise = row.get("reference", "")
        hypothesis = row.get("predicted", "")
        scores = classifier({"text": premise, "text_pair": hypothesis})[0]
        score_map = {s["label"].lower(): s["score"] for s in scores}
        results.append({
            "entailment": score_map.get("entailment", 0.0),
            "neutral": score_map.get("neutral", 0.0),
            "contradiction": score_map.get("contradiction", 0.0),
        })

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["entailment", "neutral", "contradiction"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved NLI faithfulness scores to {args.output}")


if __name__ == "__main__":
    main()
